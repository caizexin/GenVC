from configs.genVC_train_configs import GPTTrainerConfig
from trainers.hifigan_trainer import HiFiGANTrainer
from layers.stream_generator import init_stream_support
import torch
import argparse

init_stream_support()

@torch.inference_mode()
def model_init(checkpoint_path, device):
    ckpt_states = torch.load(checkpoint_path, map_location="cpu")
    config = GPTTrainerConfig().new_from_dict(ckpt_states['config'])
    # config.new_from_dict(ckpt_states['config'])
    config.use_wandb = False
    config.acoustic_dvae_checkpoint = None
    config.content_dvae_checkpoint = None
    config.model_args.gpt_checkpoint = None
    config.model_args.hifigan_checkpoint = None
    config.content_dvae_checkpoint = None
    config.is_inference = True
    model = HiFiGANTrainer.init_from_config(config)
    model.load_state_dict(ckpt_states['model'], strict=False)
    model.gpt.eval()
    model.gpt.to(device)
    model.content_dvae.eval()
    model.content_dvae.to(device)
    model.hifigan.cuda()
    model.hifigan.to(device)
    model.content_extractor.model.eval()
    model.content_extractor.model.to(device)
    model.gpt.init_gpt_for_inference()
    
    print("Model initialized")
    return model, config

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_path", type=str, default="pre_train/genVC_libritts.pth")
    args.add_argument("--device", type=str, default="cuda")
    args = args.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != args.device and args.device == "cuda":
        print("CUDA is not available, using CPU instead")
    model, cfg = model_init(args.model_path, device)