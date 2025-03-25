from dataclasses import dataclass, field
from configs.base_configs import BaseTrainingConfig, BaseAudioConfig
from typing import Tuple, List

@dataclass
class VAEConfig(BaseTrainingConfig):
    # training config
    lr: float = 1e-4
    opt_betas: List[float] = field(default_factory=lambda: [0.9, 0.997])
    #audio configuration
    audio: BaseAudioConfig = field(default_factory=BaseAudioConfig)
    feat_type: str = "Mel-spectrogram"
    mel_norm_file: str = None
    contentvec_model_path: str = None
    warmup_steps: int = 1000

    # dataset
    batch_size: int = 8
    eval_batch_size: int = 8
    num_loader_workers: int = 4
    num_eval_loader_workers: int = 4
    max_wav_len: int = 16384
    train_metafile: str = "data/train.txt"
    test_metafile: str = "data/test.txt"

    # logging and saving
    epochs: int = 1000
    grad_clip_norm: float = 0.5
    log_interval: int = 100
    save_interval: int = 1000
    eval_interval: int = 5000
    save_dir: str = "exp/dvae"
    use_wandb: bool = False
    vae_checkpoint: str = None
    wandb_project: str = "vae"
    wandb_run_name: str = "vae"

    # model config
    num_channels: int = 80
    num_tokens: int = 256
    codebook_dim: int = 512
    hidden_dim: int = 64
    num_resnet_blocks: int = 1
    kernel_size: int = 3
    num_layers: int = 2