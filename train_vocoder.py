from configs.genVC_train_configs import GPTArgs, genVCAudioConfig, GPTTrainerConfig
from trainers.hifigan_trainer import HiFiGANTrainer
from configs.vocoder_configs import BaseVocoderConfig
from configs.vae_config import VAEConfig
from configs.base_configs import BaseAudioConfig
from trainer import Trainer, TrainerArgs

MEL_NORM_FILE = 'pre_trained/mel_stats.pth'
DVAE_CHECKPOINT = 'pre_trained/acoustic_dvae.pth'
CONTENT_DVAE_CHECKPOINT = 'pre_trained/content_dvae.pth'
CONTENTVEC_MODEL_PATH = 'pre_trained/contentVec.pt'
GPT_CHECKPOINT = 'pre_trained/gpt.pth'
VOCODER_CHECKPOINT = None

# copy the config from train_audio_dvae.py
acousticDVAE_audio_config = BaseAudioConfig(dvae_sample_rate=24000)
acousticDVAE_config = VAEConfig(
    audio=acousticDVAE_audio_config,
    mel_norm_file=MEL_NORM_FILE,
    num_channels=80,
    num_tokens=1024,
    codebook_dim=512,
    hidden_dim=512,
    num_resnet_blocks=3,
    kernel_size=3,
    num_layers=2,
)

# copy the config from train_content_dvae.py
contentDVAE_audio_config = BaseAudioConfig(dvae_sample_rate=16000)
contentDVAE_config = VAEConfig(
    audio=contentDVAE_audio_config,
    mel_norm_file=MEL_NORM_FILE,
    num_channels=256,
    num_tokens=256,
    codebook_dim=512,
    hidden_dim=512,
    num_resnet_blocks=3,
    kernel_size=3,
    num_layers=2,
)

model_args = GPTArgs(
    mel_norm_file=MEL_NORM_FILE,
    gpt_num_audio_tokens=1026,
    gpt_start_audio_token=1024,
    gpt_stop_audio_token=1025,
    gpt_start_text_token=256,
    gpt_stop_text_token=257,
    gpt_number_text_tokens=258,
    gpt_fix_condition_embeddings=True,
    gpt_use_masking_gt_prompt_approach=True,
    min_text_length=8, # 8 tokens = 0.64 seconds for 20ms frame rate,
    max_text_length=8,
    gpt_n_heads=4,
    gpt_checkpoint=GPT_CHECKPOINT,
    hifigan_checkpoint=VOCODER_CHECKPOINT,
)

audio_config = genVCAudioConfig()
vocoder_config = BaseVocoderConfig()

config = GPTTrainerConfig(
    contentvec_model_path=CONTENTVEC_MODEL_PATH,
    acoustic_dvae_checkpoint=DVAE_CHECKPOINT,
    content_dvae_checkpoint=CONTENT_DVAE_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
    model_args=model_args,
    audio=audio_config,
    content_dvae_config=contentDVAE_config,
    acoustic_dvae_config=acousticDVAE_config,
    vocoder_config=vocoder_config,
    batch_size=64,
    eval_batch_size=64,
    num_loader_workers=24,
    epochs=50,
    print_step=50,
    plot_step=500,
    log_model_step=100,
    save_step=5000,
    print_eval=False,
    save_n_checkpoints=2,
    save_checkpoints=True,
    run_name="hifi-gan",
    optimizer="AdamW",
    output_path="exp/HiFiGAN_LibriTTS",
    optimizer_wd_only_on_weights=True,
    lr=2e-4,
    optimizer_params={"betas": [0.8, 0.99], "eps": 1e-8, "weight_decay": 1e-6},
    weight_decay=1e-6,
    warmup_steps=1000,
    max_grad_norm=1.0,
    train_metafile='metafiles/libritts/train.txt',
    test_metafile='metafiles/libritts/test.txt',
    use_wandb=True,
    wandb_project='hifi-gan',
    wandb_run_name='libritts',
    
)

if __name__ == '__main__':
    # currently, we don't support resuming training from a Coqui Vocoder Trainer checkpoint for the HiFi-GAN Trainer
    # please use hifigan_checkpoint to specify the path to the HiFi-GAN checkpoint instead
    restore_path = None
    model = HiFiGANTrainer.init_from_config(config)
    trainer_args = TrainerArgs(restore_path=restore_path)
    trainer = Trainer(
            trainer_args,
            config,
            model=model,
            output_path=config.output_path,
    )

    trainer.fit()
