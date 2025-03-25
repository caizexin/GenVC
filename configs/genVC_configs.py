from dataclasses import dataclass, field
from typing import List

from coqpit import Coqpit

from configs.base_configs import BaseAudioConfig, BaseTrainingConfig

@dataclass
class BaseVCConfig(BaseTrainingConfig):
    """Shared parameters among all the tts models.

    Args:

        audio (BaseAudioConfig):
            Audio processor config object instance.

        loss_masking (bool):
            enable / disable masking loss values against padded segments of samples in a batch.

        min_text_len (int):
            Minimum length of input content token to be used. All shorter samples will be ignored. Defaults to 0.

        max_text_len (int):
            Maximum length of input content token to be used. All longer samples will be ignored. Defaults to float("inf").

        min_audio_len (int):
            Minimum length of input audio to be used. All shorter samples will be ignored. Defaults to 0.

        max_audio_len (int):
            Maximum length of input audio to be used. All longer samples will be ignored. The maximum length in the
            dataset defines the VRAM used in the training. Hence, pay attention to this value if you encounter an
            OOM error in training. Defaults to float("inf").

        precompute_num_workers (int):
            Number of workers to precompute features. Defaults to 0.

        use_noise_augment (bool):
            Augment the input audio with random noise.

        start_by_longest (bool):
            If True, the data loader will start loading the longest batch first. It is useful for checking OOM issues.
            Defaults to False.

        shuffle (bool):
            If True, the data loader will shuffle the dataset when there is not sampler defined. Defaults to True.

        drop_last (bool):
            If True, the data loader will drop the last batch if it is not complete. It helps to prevent
            issues that emerge from the partial batch statistics. Defaults to True.

        optimizer (str):
            Optimizer used for the training. Set one from `torch.optim.Optimizer`.
            Defaults to ``.

        optimizer_params (dict):
            Optimizer kwargs. Defaults to `{"betas": [0.8, 0.99], "weight_decay": 0.0}`

        lr_scheduler (str):
            Learning rate scheduler for the training. Use one from `torch.optim.Scheduler` schedulers. Defaults to ``.

        lr_scheduler_params (dict):
            Parameters for the generator learning rate scheduler. Defaults to `{"warmup": 4000}`.

        use_speaker_weighted_sampler (bool):
            Enable / Disable the batch balancer by speaker. Defaults to ```False```.

        speaker_weighted_sampler_alpha (float):
            Number that control the influence of the speaker sampler weights. Defaults to ```1.0```.

        use_language_weighted_sampler (bool):
            Enable / Disable the batch balancer by language. Defaults to ```False```.

        language_weighted_sampler_alpha (float):
            Number that control the influence of the language sampler weights. Defaults to ```1.0```.
    """

    audio: BaseAudioConfig = field(default_factory=BaseAudioConfig)
    # training params
    loss_masking: bool = None
    # dataloading
    min_audio_len: int = 1
    max_audio_len: int = float("inf")
    min_text_len: int = 100 # 100 tokens, 2 seconds for 20ms frame rate,
    max_text_len: int = 300 # 300 tokens, 6 seconds for 20ms frame rate
    precompute_num_workers: int = 0
    use_noise_augment: bool = False
    start_by_longest: bool = False
    shuffle: bool = False
    drop_last: bool = False
    # optimizer
    optimizer: str = "radam"
    optimizer_params: dict = None
    # scheduler
    lr_scheduler: str = None
    lr_scheduler_params: dict = field(default_factory=lambda: {})

@dataclass
class GenVCModelArgs(Coqpit):
    """A dataclass to represent XTTS model arguments that define the model structure.

    Args:
        gpt_batch_size (int): The size of the auto-regressive batch.
        kv_cache (bool, optional): Whether to use the kv_cache. Defaults to True.
        gpt_checkpoint (str, optional): The checkpoint for the autoregressive model. Defaults to None.
        decoder_checkpoint (str, optional): The checkpoint for the HiFiGAN model. Defaults to None.
        For GPT model:
        gpt_max_audio_tokens (int, optional): The maximum mel tokens for the autoregressive model. Defaults to 604.
        gpt_max_text_tokens (int, optional): The maximum text tokens for the autoregressive model. Defaults to 402.
        gpt_max_prompt_tokens (int, optional): The maximum prompt tokens or the autoregressive model. Defaults to 70.
        gpt_layers (int, optional): The number of layers for the autoregressive model. Defaults to 30.
        gpt_n_model_channels (int, optional): The model dimension for the autoregressive model. Defaults to 1024.
        gpt_n_heads (int, optional): The number of heads for the autoregressive model. Defaults to 16.
        gpt_number_text_tokens (int, optional): The number of text tokens for the autoregressive model. Defaults to 255.
        gpt_start_text_token (int, optional): The start text token for the autoregressive model. Defaults to 255.
        gpt_checkpointing (bool, optional): Whether to use checkpointing for the autoregressive model. Defaults to False.
        gpt_train_solo_embeddings (bool, optional): Whether to train embeddings for the autoregressive model. Defaults to False.
        gpt_code_stride_len (int, optional): The hop_size of dvae and consequently of the gpt output. Defaults to 1024.
        gpt_use_masking_gt_prompt_approach (bool, optional):  If True, it will use ground truth as prompt and it will mask the loss to avoid repetition. Defaults to True.
    """

    gpt_batch_size: int = 1
    kv_cache: bool = True
    gpt_checkpoint: str = None
    decoder_checkpoint: str = None

    # XTTS GPT Encoder params
    gpt_max_audio_tokens: int = 605
    gpt_max_text_tokens: int = 402
    gpt_max_prompt_tokens: int = 70
    gpt_layers: int = 30
    gpt_n_model_channels: int = 1024
    gpt_n_heads: int = 16
    gpt_number_text_tokens: int = None
    gpt_start_text_token: int = None
    gpt_stop_text_token: int = None
    gpt_num_audio_tokens: int = 1026
    gpt_start_audio_token: int = 1024
    gpt_stop_audio_token: int = 1025
    gpt_code_stride_len: int = 1024

    # constants
    duration_const: int = 102400

    min_conditioning_length: int = 66150
    max_conditioning_length: int = 132300
    gpt_loss_text_ce_weight: float = 0.01
    gpt_loss_mel_ce_weight: float = 1.0
    gpt_num_audio_tokens: int = 1026
    debug_loading_failures: bool = False
    max_text_length: int = 300 # 400 tokens, 8 seconds for 20ms frame rate
    mel_norm_file: str = "https://coqui.gateway.scarf.sh/v0.14.0_models/mel_norms.pth"
    dvae_checkpoint: str = ""
    content_dvae_checkpoint: str = ""
    gpt_checkpoint: str = ""  # if defined it will replace the gpt weights on xtts model
    vocoder: str = ""  # overide vocoder key on the config to avoid json write issues
    gpt_use_masking_gt_prompt_approach: bool = False
    gpt_fix_condition_embeddings: bool = False




