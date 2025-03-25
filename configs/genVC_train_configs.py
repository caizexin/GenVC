from dataclasses import dataclass, field
from configs.base_configs import BaseAudioConfig
from configs.genVC_configs import GenVCModelArgs, BaseVCConfig
from configs.vae_config import VAEConfig
from configs.vocoder_configs import BaseVocoderConfig
from typing import Dict, List, Tuple, Union

@dataclass
class genVCAudioConfig(BaseAudioConfig):
    dvae_sample_rate: int = 24000
    sample_rate: int = 24000
    output_sample_rate: int = 24000
    # content_sample_rate: int = 16000

@dataclass
class GPTArgs(GenVCModelArgs):
    min_text_length: int = 100
    max_text_length: int = 300
    min_conditioning_length: int = 72000
    max_conditioning_length: int = 144000
    gpt_loss_text_ce_weight: float = 0.01
    gpt_loss_mel_ce_weight: float = 1.0
    gpt_num_audio_tokens: int = 1026
    debug_loading_failures: bool = False
    gpt_content_dim: int = 256
    dvae_checkpoint: str = ""
    gpt_checkpoint: str = ""  
    hifigan_checkpoint: str = ""
    gpt_fix_condition_embeddings: bool = False

@dataclass
class GPTTrainerConfig(BaseVCConfig):
    lr: Union[float,List[float]] = 5e-06
    training_seed: int = 1
    optimizer_wd_only_on_weights: bool = True
    weighted_loss_attrs: dict = field(default_factory=lambda: {})
    weighted_loss_multipliers: dict = field(default_factory=lambda: {})
    model_args: GPTArgs = field(default_factory=GPTArgs)
    acoustic_dvae_config: VAEConfig = field(default_factory=VAEConfig)
    content_dvae_config: VAEConfig = field(default_factory=VAEConfig)
    vocoder_config: BaseVocoderConfig = field(default_factory=BaseVocoderConfig)
    model_dir: str = None
    epochs: int = 200
    weight_decay: float = 0.0
    warmup_steps: int = 1000
    lr_scheduler: str = "cosine"
    lr_decay: float = 0.98
    max_grad_norm: float = 1.0
    use_ddp: bool = False
    use_cuda: bool = True
    use_accelerate: bool = False
    seed: int = 1994
    is_inference: bool = False

    # logging and saving
    log_interval: int = 100
    save_interval: int = 1000
    eval_interval: int = 1000
    save_dir: str = "exp/gpt"
    resume_checkpoint: str = None
    
    # wandb
    use_wandb: bool = True
    wandb_project: str = "train_gpt_vc"
    wandb_run_name: str = "libritts"

    # dataset
    batch_size: int = 8
    eval_batch_size: int = 4
    num_workers: int = 0
    train_metafile: str = 'metafiles/libritts_train.txt'
    test_metafile: str = 'metafiles/libritts_test.txt'
    text_frame_rate: float = 0.02
    
    # inference
    temperature: float = 0.85
    length_penalty: float = 1.0
    repetition_penalty: float = 2.0
    top_k: int = 15
    top_p: float = 0.85
    num_gpt_outputs: int = 1

    # cloning
    gpt_cond_len: int = 12
    gpt_cond_chunk_len: int = 4
    max_ref_len: int = 10
    sound_norm_refs: bool = False
    acoustic_dvae_checkpoint: str = ""
    content_dvae_checkpoint: str = ""
    contentvec_model_path: str = "pretrained_models/contentVec.pth"
