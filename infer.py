from inference.model_init import model_init
from inference.inference_utils import synthesize_utt, synthesize_utt_streaming
from utils import load_audio
import torchaudio
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='pre_trained/GenVC_large.pth')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--src_wav', type=str, default='samples/EF4_ENG_0112_1.wav')
    parser.add_argument('--ref_audio', type=str, default='samples/EM1_ENG_0037_1.wav')
    parser.add_argument('--output_path', type=str, default='samples/converted.wav')
    parser.add_argument('--top_k', type=int, default=15)
    parser.add_argument('--streaming', action='store_true')
    args = parser.parse_args()

    model, config = model_init(args.model_path, args.device)

    # top_k is one of the important hyperparameters for inference, so you can tune it to get better results
    # for streaming inference, greedy decoding is preferred, you can set top_k to 1
    model.config.top_k = args.top_k
    src_wav = load_audio(args.src_wav, model.content_sample_rate)
    ref_audio = load_audio(args.ref_audio, model.config.audio.sample_rate)

    if args.streaming:
        # for accurate latency measurements, please warm up the model before inference, for example by running a dummy inference
        # warmup_times = 3
        # for _ in range(warmup_times):
        #     synthesize_utt_streaming(model, src_wav, tgt_audio)
        # The performance would be better with a causal vocoder, which will be available in the future
        pre_audio = synthesize_utt_streaming(model, src_wav, ref_audio)
    else:
        pre_audio = synthesize_utt(model, src_wav, ref_audio)

    torchaudio.save(args.output_path, pre_audio.unsqueeze(0).detach().cpu(), config.audio.sample_rate)

