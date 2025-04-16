import torch
import time

@torch.inference_mode()
def handle_chunks(wav_gen, wav_gen_prev, wav_overlap, overlap_len=1024):
    """Handle chunk formatting in streaming mode"""
    wav_chunk = wav_gen[:-overlap_len]
    if wav_overlap is not None:
        # cross fade the overlap section
        if overlap_len > len(wav_chunk):
            wav_chunk = wav_gen[-overlap_len:]
            return wav_chunk, wav_gen, None
        else:
            crossfade_wav = wav_chunk[:overlap_len]
            crossfade_wav = crossfade_wav * torch.linspace(0.0, 1.0, overlap_len).to(crossfade_wav.device)
            wav_chunk[:overlap_len] = wav_overlap * torch.linspace(1.0, 0.0, overlap_len).to(wav_overlap.device)
            wav_chunk[:overlap_len] += crossfade_wav

    wav_overlap = wav_gen[-overlap_len:]
    wav_gen_prev = wav_gen
    return wav_chunk, wav_gen_prev, wav_overlap

@torch.inference_mode()
def synthesize_utt(
    genVC_mdl, 
    src_wav, 
    tgt_audio, 
    seg_len=6.0):
    """Synthesize audio in chunks, used for non-streaming mode
    The concatenation is performed at the latent feature level"""
    wav_gen_prev, wav_overlap = None, None
    total_wavlen = src_wav.shape[-1]
    pred_audios = []
    min_chunk_duration = int(0.32 * genVC_mdl.content_sample_rate)

    src_wav = src_wav.to(genVC_mdl.device)
    seg_len = int(seg_len * genVC_mdl.content_sample_rate)
    # get the conditioning latent
    tgt_audio = tgt_audio.to(genVC_mdl.device)
    cond_latent = genVC_mdl.get_gpt_cond_latents(tgt_audio, genVC_mdl.config.audio.sample_rate)
    final_latents = []

    for i in range(0, total_wavlen, seg_len):
        seg_end = i+seg_len if i+seg_len < total_wavlen else total_wavlen
        if seg_end == total_wavlen:
            src_wav_seg = src_wav[:, i:]
            if src_wav_seg.shape[-1] < min_chunk_duration:
                src_wav_seg = torch.nn.functional.pad(src_wav_seg, (0, min_chunk_duration-src_wav_seg.shape[-1]), "constant", 0)
        else:
            src_wav_seg = src_wav[:, i:i+seg_len]

        content_feat = genVC_mdl.content_extractor.extract_content_features(src_wav_seg)
        content_codes = genVC_mdl.content_dvae.get_codebook_indices(content_feat.transpose(1, 2))
        
        gen_codes = genVC_mdl.gpt.generate(
            cond_latent,
            content_codes,
            do_sample=True,
            top_p=genVC_mdl.config.top_p,
            top_k=genVC_mdl.config.top_k,
            temperature=genVC_mdl.config.temperature,
            num_beams=1,
            length_penalty=genVC_mdl.config.length_penalty,
            repetition_penalty=genVC_mdl.config.repetition_penalty,
            output_attentions=False,
        )[0]

        gen_codes = gen_codes[(gen_codes!=genVC_mdl.gpt.stop_audio_token).nonzero().squeeze()]
        expected_output_len = torch.tensor([gen_codes.shape[-1] * genVC_mdl.config.model_args.gpt_code_stride_len], device=genVC_mdl.device)
        content_len = torch.tensor([content_codes.shape[-1]], device=genVC_mdl.device)
        acoustic_latents = genVC_mdl.gpt(content_codes,
                                    content_len,
                                    gen_codes.unsqueeze(0),
                                    expected_output_len,
                                    cond_latents=cond_latent,
                                    return_latent=True)
        final_latents.append(acoustic_latents)
    
    # concatenate the latents
    final_latents = torch.cat(final_latents, dim=1)
    mel_input = torch.nn.functional.interpolate(
        final_latents.transpose(1, 2),
        scale_factor=[genVC_mdl.hifigan_scale_factor],
        mode="linear",
    ).squeeze(1)

    synthesized_audio = genVC_mdl.hifigan(mel_input)

    return synthesized_audio[0].squeeze()

@torch.inference_mode()
def synthesize_utt_chunked(
    genVC_mdl, 
    src_wav, 
    tgt_audio, 
    seg_len=6.0):
    """Synthesize audio in chunks, used for non-streaming mode
    The concatenation is performed at the waveform level"""
    wav_gen_prev, wav_overlap = None, None
    total_wavlen = src_wav.shape[-1]
    pred_audios = []
    min_chunk_duration = 0.32 * genVC_mdl.content_sample_rate

    src_wav = src_wav.to(genVC_mdl.device)
    seg_len = int(seg_len * genVC_mdl.content_sample_rate)
    # get the conditioning latent
    tgt_audio = tgt_audio.to(genVC_mdl.device)
    cond_latent = genVC_mdl.get_gpt_cond_latents(tgt_audio, genVC_mdl.config.audio.sample_rate)

    for i in range(0, total_wavlen, seg_len):
        seg_end = i+seg_len if i+seg_len < total_wavlen else total_wavlen
        if seg_end == total_wavlen:
            src_wav_seg = src_wav[:, i:]
            if src_wav_seg.shape[-1] < min_chunk_duration:
                src_wav_seg = torch.nn.functional.pad(src_wav_seg, (0, min_chunk_duration-src_wav_seg.shape[-1]), "constant", 0)
        else:
            src_wav_seg = src_wav[:, i:i+seg_len]
        audio_pred = genVC_mdl.inference(
            src_wav_seg, 
            cond_latent,
            top_p=genVC_mdl.config.top_p,
            top_k=genVC_mdl.config.top_k,
            temperature=genVC_mdl.config.temperature,
            length_penalty=genVC_mdl.config.length_penalty,
            repetition_penalty=genVC_mdl.config.repetition_penalty)
        
        wav_chunk, wav_gen_prev, wav_overlap = handle_chunks(
            audio_pred.squeeze(), wav_gen_prev, wav_overlap, 1024)
        pred_audios.append(wav_chunk)
    
    synthesized_audio = torch.cat(pred_audios, dim=-1)

    return synthesized_audio

@torch.inference_mode()
def synthesize_utt_streaming(
    genVC_mdl, 
    src_wav, 
    tgt_audio, 
    seg_len=6.0,
    stream_chunk_size=8):

    wav_gen_prev, wav_overlap = None, None
    total_wavlen = src_wav.shape[-1]
    pred_audios = []
    min_chunk_duration = 0.32 * genVC_mdl.content_sample_rate

    begin_time = time.time()

    src_wav = src_wav.to(genVC_mdl.device)
    seg_len = int(seg_len * genVC_mdl.content_sample_rate)
    # get the conditioning latent
    tgt_audio = tgt_audio.to(genVC_mdl.device)
    cond_latent = genVC_mdl.get_gpt_cond_latents(tgt_audio, genVC_mdl.config.audio.sample_rate)
    is_begin = True
    
    for i in range(0, total_wavlen, seg_len):
        seg_end = i+seg_len if i+seg_len < total_wavlen else total_wavlen
        if seg_end == total_wavlen:
            src_wav_seg = src_wav[:, i:]
            if src_wav_seg.shape[-1] < min_chunk_duration:
                src_wav_seg = torch.nn.functional.pad(src_wav_seg, (0, min_chunk_duration-src_wav_seg.shape[-1]), "constant", 0)
        else:
            src_wav_seg = src_wav[:, i:i+seg_len]

        content_feat = genVC_mdl.content_extractor.extract_content_features(src_wav_seg)
        content_codes = genVC_mdl.content_dvae.get_codebook_indices(content_feat.transpose(1, 2))
        gpt_inputs = genVC_mdl.gpt.compute_embeddings(cond_latent, content_codes)

        gpt_generator = genVC_mdl.gpt.get_generator(
            fake_inputs=gpt_inputs,
            top_p=genVC_mdl.config.top_p,
            top_k=genVC_mdl.config.top_k,
            temperature=genVC_mdl.config.temperature,
            length_penalty=genVC_mdl.config.length_penalty,
            repetition_penalty=genVC_mdl.config.repetition_penalty,
            do_sample=True,
            num_beams=1,
            num_return_sequences=1,
            output_attentions=False,
            output_hidden_states=True,
        )

        last_tokens = []
        all_latents = []
        is_end = False
        while not is_end:
            try:
                x, latent = next(gpt_generator)
                last_tokens += [x]
                all_latents += [latent]
            except StopIteration:
                is_end = True

            if is_end or (stream_chunk_size > 0 and len(last_tokens) >= stream_chunk_size):
                acoustic_latents = torch.cat(all_latents, dim=0)[None, :]
                mel_input = torch.nn.functional.interpolate(
                    acoustic_latents.transpose(1, 2),
                    scale_factor=[genVC_mdl.hifigan_scale_factor],
                    mode="linear",
                ).squeeze(1)
                audio_pred = genVC_mdl.hifigan.forward(mel_input)
                wav_chunk, wav_gen_prev, wav_overlap = handle_chunks(
                    audio_pred.squeeze(), wav_gen_prev, wav_overlap, 1024)
                pred_audios.append(wav_chunk)
                last_tokens = []
                all_latents = []
                if is_begin:
                    is_begin = False
                    latency = time.time() - begin_time
                    print(f"Latency: {latency:.3f}s")
    
    synthesized_audio = torch.cat(pred_audios, dim=-1)
    processed_time = time.time() - begin_time
    real_time_factor = processed_time / (total_wavlen / genVC_mdl.content_sample_rate)
    print(f"Real-time factor: {real_time_factor:.3f}")
    return synthesized_audio
