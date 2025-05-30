# ported from: https://github.com/coqui-ai/TTS
# ported from: https://github.com/neonbjb/tortoise-tts

import functools
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config
from torchmetrics.classification import MulticlassAccuracy
from layers.gpt_inference import GPT2InferenceModel
from layers.perceiver_encoder import PerceiverResampler
from utils import get_mask_from_lengths

def null_position_embeddings(range, dim):
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=0.02, relative=False):
        super().__init__()
        # nn.Embedding
        self.emb = torch.nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)
        self.relative = relative
        self.seq_len = seq_len

    def forward(self, x):
        sl = x.shape[1]
        if self.relative:
            start = random.randint(sl, self.seq_len) - sl
            return self.emb(torch.arange(start, start + sl, device=x.device))
        else:
            return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)

def build_hf_gpt_transformer(
    layers,
    model_dim,
    heads,
    max_mel_seq_len,
    max_text_seq_len,
    max_prompt_len,
    checkpointing,
):
    """
    GPT-2 implemented by the HuggingFace library.
    """
    from transformers import GPT2Config, GPT2Model

    gpt_config = GPT2Config(
        vocab_size=256,  # Unused.
        n_positions=max_mel_seq_len + max_text_seq_len + max_prompt_len,
        n_ctx=max_mel_seq_len + max_text_seq_len + max_prompt_len,
        n_embd=model_dim,
        n_layer=layers,
        n_head=heads,
        gradient_checkpointing=checkpointing,
        use_cache=not checkpointing,
    )
    gpt = GPT2Model(gpt_config)
    # Override the built in positional embeddings
    del gpt.wpe
    gpt.wpe = functools.partial(null_position_embeddings, dim=model_dim)
    # Built-in token embeddings are unused.
    del gpt.wte

    mel_pos_emb = (
        LearnedPositionEmbeddings(max_mel_seq_len, model_dim)
        if max_mel_seq_len != -1
        else functools.partial(null_position_embeddings, dim=model_dim)
    )
    text_pos_emb = (
        LearnedPositionEmbeddings(max_text_seq_len, model_dim)
        if max_mel_seq_len != -1
        else functools.partial(null_position_embeddings, dim=model_dim)
    )
    # gpt = torch.compile(gpt, mode="reduce-overhead", fullgraph=True)
    return gpt, mel_pos_emb, text_pos_emb, None, None


class GPT(nn.Module):
    def __init__(
        self,
        start_text_token=256,
        stop_text_token=257,
        layers=8,
        model_dim=512,
        heads=8,
        max_text_tokens=120,
        max_mel_tokens=250,
        max_prompt_tokens=70,
        max_conditioning_inputs=1,
        code_stride_len=1024,
        number_text_tokens=258,
        num_audio_tokens=1026,
        start_audio_token=1024,
        stop_audio_token=1025,
        train_solo_embeddings=False,
        checkpointing=False,
        average_conditioning_embeddings=False,
        fix_condition_embeddings=False,
        label_smoothing=0.0,
        perceiver_cond_length_compression=256,
    ):
        """
        Args:

        """
        super().__init__()

        self.label_smoothing = label_smoothing
        self.number_text_tokens = number_text_tokens
        self.start_text_token = start_text_token
        self.stop_text_token = stop_text_token
        self.num_audio_tokens = num_audio_tokens
        self.start_audio_token = start_audio_token
        self.stop_audio_token = stop_audio_token
        self.start_prompt_token = start_audio_token
        self.stop_prompt_token = stop_audio_token
        self.layers = layers
        self.heads = heads
        self.model_dim = model_dim
        self.fix_condition_embeddings = fix_condition_embeddings
        self.max_conditioning_inputs = max_conditioning_inputs
        self.max_gen_mel_tokens = max_mel_tokens - self.max_conditioning_inputs - 2
        self.max_mel_tokens = -1 if max_mel_tokens == -1 else max_mel_tokens + 2 + self.max_conditioning_inputs
        self.max_text_tokens = -1 if max_text_tokens == -1 else max_text_tokens + 2
        self.max_prompt_tokens = max_prompt_tokens
        self.code_stride_len = code_stride_len
        
        self.conditioning_dropout = nn.Dropout1d(0.1)
        self.average_conditioning_embeddings = average_conditioning_embeddings
        self.perceiver_cond_length_compression = perceiver_cond_length_compression

        self.text_embedding = nn.Embedding(self.number_text_tokens, model_dim)
        self.mel_embedding = nn.Embedding(self.num_audio_tokens, model_dim)

        (
            self.gpt,
            self.mel_pos_embedding,
            self.text_pos_embedding,
            self.mel_layer_pos_embedding,
            self.text_layer_pos_embedding,
        ) = build_hf_gpt_transformer(
            layers,
            model_dim,
            heads,
            self.max_mel_tokens,
            self.max_text_tokens,
            self.max_prompt_tokens,
            checkpointing,
        )
        if train_solo_embeddings:
            self.mel_solo_embedding = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02, requires_grad=True)
            self.text_solo_embedding = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02, requires_grad=True)
        else:
            self.mel_solo_embedding = 0
            self.text_solo_embedding = 0

        self.accuracy_metric = MulticlassAccuracy(
            num_audio_tokens,
            top_k=10,
            average="micro",
            multidim_average="global",
            ignore_index=-1,
        )

        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.number_text_tokens)
        self.mel_head = nn.Linear(model_dim, self.num_audio_tokens)

        # We haven't included the hyper-parameters of the perceiver for configuration purposes.
        self.conditioning_perceiver = PerceiverResampler(
            dim=model_dim,
            depth=4,
            dim_context=80,
            num_latents=32,
            dim_head=64,
            heads=8,
            ff_mult=4,
            use_flash_attn=False,
        )

    def get_grad_norm_parameter_groups(self):
        return {
            "conditioning_perceiver": list(self.conditioning_perceiver.parameters()),
            "gpt": list(self.gpt.parameters()),
            "heads": list(self.text_head.parameters()) + list(self.mel_head.parameters()),
        }

    def init_gpt_for_inference(self, kv_cache=True, use_deepspeed=False):
        seq_length = self.max_prompt_tokens + self.max_mel_tokens + self.max_text_tokens + 1
        gpt_config = GPT2Config(
            vocab_size=self.max_mel_tokens,
            n_positions=seq_length,
            n_ctx=seq_length,
            n_embd=self.model_dim,
            n_layer=self.layers,
            n_head=self.heads,
            gradient_checkpointing=False,
            use_cache=True,
        )
        self.gpt_inference = GPT2InferenceModel(
            gpt_config,
            self.gpt,
            self.mel_pos_embedding,
            self.mel_embedding,
            self.final_norm,
            self.mel_head,
            kv_cache=kv_cache,
        )
        self.gpt.wte = self.mel_embedding

        if use_deepspeed:
            import deepspeed

            self.ds_engine = deepspeed.init_inference(
                model=self.gpt_inference.half(),  # Transformers models
                mp_size=1,  # Number of GPU
                dtype=torch.float32,  # desired data type of output
                replace_method="auto",  # Lets DS autmatically identify the layer to replace
                replace_with_kernel_inject=True,  # replace the model with the kernel injector
            )
            self.gpt_inference = self.ds_engine.module.eval()

    def set_inputs_and_targets(self, input, start_token, stop_token):
        inp = F.pad(input, (1, 0), value=start_token)
        tar = F.pad(input, (0, 1), value=stop_token)
        return inp, tar

    def set_mel_padding(self, mel_input_tokens, code_lengths):
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with stop_audio_token in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        # Set padding areas within MEL (currently it is coded with the MEL code for <zero>).
        for b in range(len(code_lengths)):
            actual_end = code_lengths[b]
            if actual_end < mel_input_tokens.shape[-1]:
                mel_input_tokens[b, actual_end:] = self.stop_audio_token
        return mel_input_tokens
    
    def set_text_padding(self, text_input_tokens, code_lengths):
        """
        Given content tokens that are derived from a padded semantic feature and the actual lengths of each batch element, reformats the tokens with stop_text_token in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        # Set padding areas within MEL (currently it is coded with the MEL code for <zero>).
        for b in range(len(code_lengths)):
            actual_end = code_lengths[b]
            if actual_end < text_input_tokens.shape[-1]:
                text_input_tokens[b, actual_end:] = self.stop_text_token
        return text_input_tokens

    def get_logits(
        self,
        first_inputs,
        first_head,
        second_inputs=None,
        second_head=None,
        prompt=None,
        get_attns=False,
        return_latent=False,
        attn_mask_cond=None,
        attn_mask_text=None,
        attn_mask_mel=None,
    ):
        if prompt is not None:
            offset = prompt.shape[1]
            if second_inputs is not None:
                emb = torch.cat([prompt, first_inputs, second_inputs], dim=1)
            else:
                emb = torch.cat([prompt, first_inputs], dim=1)

        # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        attn_mask = None
        if attn_mask_text is not None:
            attn_mask = torch.cat([attn_mask_text, attn_mask_mel], dim=1)
            if prompt is not None:
                attn_mask_cond = torch.ones(prompt.shape[0], offset, dtype=torch.bool, device=emb.device)
                attn_mask = torch.cat([attn_mask_cond, attn_mask], dim=1)

        gpt_out = self.gpt(
            inputs_embeds=emb,
            return_dict=True,
            output_attentions=get_attns,
            attention_mask=attn_mask,
        )

        if get_attns:
            return gpt_out.attentions

        enc = gpt_out.last_hidden_state[:, offset:]
        enc = self.final_norm(enc)

        if return_latent:
            return enc[:, : first_inputs.shape[1]], enc[:, -second_inputs.shape[1] :]

        first_logits = enc[:, : first_inputs.shape[1]]
        first_logits = first_head(first_logits)
        first_logits = first_logits.permute(0, 2, 1)
        if second_inputs is not None:
            second_logits = enc[:, -second_inputs.shape[1] :]
            second_logits = second_head(second_logits)
            second_logits = second_logits.permute(0, 2, 1)
            return first_logits, second_logits
        else:
            return first_logits

    def get_prompts(self, prompt_codes):
        """
        Create a prompt from the mel codes. This is used to condition the model on the mel codes.
        Pad the prompt with start and stop mel tokens.
        """
        prompt = prompt_codes
        if self.training:
            lengths = []
            # Compute the real prompt length based on the first encounter with the token 83 used for padding
            for i in range(prompt_codes.shape[0]):
                length = 0
                for j in range(prompt_codes.shape[1]):
                    if prompt_codes[i, j] == 83:
                        break
                    else:
                        length += 1
                lengths.append(length)

            # prompt_len = random.randint(1, 9)  # in secs
            prompt_len = 3
            prompt_len = prompt_len * 24  # in frames
            if prompt_codes.shape[-1] >= prompt_len:
                for i in range(prompt_codes.shape[0]):
                    if lengths[i] < prompt_len:
                        start = 0
                    else:
                        start = random.randint(0, lengths[i] - prompt_len)
                prompt = prompt_codes[:, start : start + prompt_len]

        # add start and stop tokens
        prompt = F.pad(prompt, (1, 0), value=self.start_prompt_token)
        prompt = F.pad(prompt, (0, 1), value=self.stop_prompt_token)
        return prompt

    def get_style_emb(self, cond_input, return_latent=False, seq_lens=None):
        """
        cond_input: (b, 80, s) or (b, 1, 80, s)
        conds: (b, 1024, s)
        """
        conds = None
        if not return_latent:
            if cond_input.ndim == 4:
                cond_input = cond_input.squeeze(1)
            encoder_mask = None
            percerver_mask = None
            if seq_lens is not None:
                max_seq_len = cond_input.shape[-1]
                batch_size = cond_input.shape[0]
                percerver_mask = get_mask_from_lengths(seq_lens, max_seq_len)
                encoder_mask = percerver_mask.unsqueeze(1).expand(-1, max_seq_len, -1).to(cond_input.device)
                percerver_mask = torch.cat([percerver_mask, torch.ones(batch_size, 32, dtype=torch.bool, device=cond_input.device)], dim=-1)
            
            conds = self.conditioning_perceiver(cond_input.permute(0, 2, 1), mask=percerver_mask).transpose(1, 2)  # (b, d, 32)
        else:
            # already computed
            conds = cond_input.unsqueeze(1)
        return conds

    def forward(
        self,
        text_inputs,
        text_lengths,
        audio_codes,
        wav_lengths,
        cond_mels=None,
        cond_lens=None,
        cond_latents=None,
        return_attentions=False,
        return_latent=False,
    ):
        """
        Forward pass that uses both text and voice in either text conditioning mode or voice conditioning mode
        (actuated by `text_first`).

        text_inputs: long tensor, (b,t)
        text_lengths: long tensor, (b,)
        mel_inputs:  long tensor, (b,m)
        wav_lengths: long tensor, (b,)
        cond_mels: MEL float tensor, (b, 1, 80,s)

        If return_attentions is specified, only logits are returned.
        If return_latent is specified, loss & logits are not computed or returned. Only the predicted latents are returned.
        """
        # ❗ FIXIT
        if self.max_conditioning_inputs == 0:
            assert cond_mels is None, " ❗ cond_mels is not None, but max_conditioning_inputs == 0"

        max_text_len = text_lengths.max()
        code_lengths = torch.ceil(wav_lengths / self.code_stride_len).long() + 3

        if cond_lens is not None:
            cond_lens = cond_lens // self.perceiver_cond_length_compression

        # If len(codes) + 3 is larger than maxiumum allowed length, we truncate the codes.
        max_mel_len = code_lengths.max()

        if max_mel_len > audio_codes.shape[-1]:
            audio_codes = F.pad(audio_codes, (0, max_mel_len - audio_codes.shape[-1]))

        # 💖 Lovely assertions
        assert (
            max_mel_len <= audio_codes.shape[-1]
        ), f" ❗ max_mel_len ({max_mel_len}) > audio_codes.shape[-1] ({audio_codes.shape[-1]})"
        assert (
            max_text_len <= text_inputs.shape[-1]
        ), f" ❗ max_text_len ({max_text_len}) > text_inputs.shape[-1] ({text_inputs.shape[-1]})"

        # Append stop token to text inputs
        text_inputs = F.pad(text_inputs[:, :max_text_len], (0, 1), value=self.stop_text_token)

        text_inputs = self.set_text_padding(text_inputs, text_lengths)

        # Append silence token to mel codes
        audio_codes = F.pad(audio_codes[:, :max_mel_len], (0, 1), value=self.stop_audio_token)

        # Pad mel codes with stop_audio_token
        audio_codes = self.set_mel_padding(
            audio_codes, code_lengths - 3
        )  # -3 to get the real code lengths without consider start and stop tokens that was not added yet

        # Build input and target tensors
        # Prepend start token to inputs and append stop token to targets
        text_inputs, text_targets = self.set_inputs_and_targets(
            text_inputs, self.start_text_token, self.stop_text_token
        )
        audio_codes, mel_targets = self.set_inputs_and_targets(
            audio_codes, self.start_audio_token, self.stop_audio_token
        )

        # Set attn_mask
        attn_mask_cond = None
        attn_mask_text = None
        attn_mask_mel = None
        if not return_latent:
            attn_mask_cond = torch.ones(
                cond_mels.shape[0],
                32,
                dtype=torch.bool,
                device=text_inputs.device,
            )
            attn_mask_text = torch.ones(
                text_inputs.shape[0],
                text_inputs.shape[1],
                dtype=torch.bool,
                device=text_inputs.device,
            )
            attn_mask_mel = torch.ones(
                audio_codes.shape[0],
                audio_codes.shape[1],
                dtype=torch.bool,
                device=audio_codes.device,
            )

            for idx, l in enumerate(text_lengths):
                attn_mask_text[idx, l + 1 :] = 0.0

            for idx, l in enumerate(code_lengths):
                attn_mask_mel[idx, l + 1 :] = 0.0

        # Compute text embeddings + positional embeddings
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)

        # Compute mel embeddings + positional embeddings
        mel_emb = self.mel_embedding(audio_codes) + self.mel_pos_embedding(audio_codes)

        # Compute speech conditioning input
        if cond_latents is None:
            if self.fix_condition_embeddings:
                with torch.no_grad():
                    cond_latents = self.get_style_emb(cond_mels, seq_lens=cond_lens).transpose(1, 2)
            else:
                cond_latents = self.get_style_emb(cond_mels, seq_lens=cond_lens).transpose(1, 2)

        # Get logits
        sub = -5  # don't ask me why 😄
        if self.training:
            sub = -1

        text_logits, mel_logits = self.get_logits(
            text_emb,
            self.text_head,
            mel_emb,
            self.mel_head,
            prompt=cond_latents,
            get_attns=return_attentions,
            return_latent=return_latent,
            attn_mask_cond=attn_mask_cond,
            attn_mask_text=attn_mask_text,
            attn_mask_mel=attn_mask_mel,
        )
        if return_latent:
            return mel_logits[:, :sub]  # sub to prevent bla.

        if return_attentions:
            return mel_logits

        # Set paddings to -1 to ignore them in loss
        for idx, l in enumerate(text_lengths):
            text_targets[idx, l + 1 :] = -1

        for idx, l in enumerate(code_lengths):
            mel_targets[idx, l + 1 :] = -1

        # check if stoptoken is in every row of mel_targets
        assert (mel_targets == self.stop_audio_token).sum() >= mel_targets.shape[
            0
        ], f" ❗ mel_targets does not contain stop token ({self.stop_audio_token}) in every row."

        # Compute losses
        loss_text = F.cross_entropy(
            text_logits, text_targets.long(), ignore_index=-1, label_smoothing=self.label_smoothing
        )
        loss_mel = F.cross_entropy(
            mel_logits, mel_targets.long(), ignore_index=-1, label_smoothing=self.label_smoothing
        )

        Top10Accuracy = self.accuracy_metric(
            mel_logits.detach(), mel_targets.long()
        )# .item() * code_lengths.sum().type(torch.float32)

        return loss_text.mean(), loss_mel.mean(), Top10Accuracy, mel_logits

    @torch.no_grad()
    def eval_sample(
        self,
        text_inputs,
        text_lens,
        cond_mels=None,
        cond_lens=None,
        cond_latents=None,
    ):
        self.init_gpt_for_inference()

        cond_latents = self.get_style_emb(cond_mels, seq_lens=cond_lens).transpose(1, 2)

        # always use sample 0 for inference
        text_input = text_inputs[0][:text_lens[0]].unsqueeze(0)
        text_input_cv = text_input.clone().to(text_inputs.device)
        cond_latent = cond_latents[0].unsqueeze(0)
        mel_code = self.generate(cond_latent, text_input)[0]
        mel_code = mel_code[(mel_code!=self.stop_audio_token).nonzero().squeeze()]

        tgt_cond_idx = random.randint(0, cond_latents.shape[0] - 1)
        cond_latent_cv = cond_latents[tgt_cond_idx].unsqueeze(0)
        cv_mel_code = self.generate(cond_latent_cv, text_input_cv)[0]
        cv_mel_code = cv_mel_code[(cv_mel_code!=self.stop_audio_token).nonzero().squeeze()]

        del self.gpt_inference
        del self.gpt.wte
        return 0, mel_code, tgt_cond_idx, cv_mel_code

    @torch.inference_mode()
    def inference(self, cond_latents, text_inputs, **generate_kwargs):
        return self.generate(cond_latents, text_inputs, **generate_kwargs)

    def compute_embeddings(
        self,
        cond_latents,
        text_inputs,
    ):
        text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)
        text_inputs = F.pad(text_inputs, (1, 0), value=self.start_text_token)
        emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)
        emb = torch.cat([cond_latents, emb], dim=1)
        self.gpt_inference.store_prefix_emb(emb)
        gpt_inputs = torch.full(
            (
                emb.shape[0],
                emb.shape[1] + 1,  # +1 for the start_audio_token
            ),
            fill_value=1,
            dtype=torch.long,
            device=text_inputs.device,
        )
        gpt_inputs[:, -1] = self.start_audio_token
        return gpt_inputs

    def generate(
        self,
        cond_latents,
        text_inputs,
        **generate_kwargs,
    ):
        gpt_inputs = self.compute_embeddings(cond_latents, text_inputs)
        gen = self.gpt_inference.generate(
            gpt_inputs,
            bos_token_id=self.start_audio_token,
            pad_token_id=self.stop_audio_token,
            eos_token_id=self.stop_audio_token,
            max_length=self.max_gen_mel_tokens + gpt_inputs.shape[-1],
            **generate_kwargs,
        )
        return gen[:, gpt_inputs.shape[1] :]

    # for streaming inference
    def get_generator(self, fake_inputs, **generate_kwargs):
        return self.gpt_inference.generate_stream(
            fake_inputs,
            bos_token_id=self.start_audio_token,
            pad_token_id=self.stop_audio_token,
            eos_token_id=self.stop_audio_token,
            max_length=self.max_gen_mel_tokens + fake_inputs.shape[-1],
            do_stream=True,
            **generate_kwargs,
        )
