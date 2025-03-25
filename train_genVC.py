from configs.genVC_train_configs import GPTArgs, genVCAudioConfig, GPTTrainerConfig
from trainers.gpt_trainer import GPTTrainer
from configs.vae_config import VAEConfig
from configs.base_configs import BaseAudioConfig
from trainer import Trainer, TrainerArgs

MEL_NORM_FILE = 'pre_trained/mel_stats.pth'
DVAE_CHECKPOINT = 'pre_trained/acoustic_dvae.pth'
CONTENT_DVAE_CHECKPOINT = 'pre_trained/content_dvae.pth'
CONTENTVEC_MODEL_PATH = 'pre_trained/contentVec.pt'
GPT_CHECKPOINT = None

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
    min_text_length=15, # 15 tokens = 1.2 seconds for 20ms frame rate
    max_text_length=100, # 100 tokens = 8 seconds for 20ms frame rate
    gpt_n_heads=4,
    gpt_checkpoint=GPT_CHECKPOINT,
)

audio_config = genVCAudioConfig()

config = GPTTrainerConfig(
    contentvec_model_path=CONTENTVEC_MODEL_PATH,
    acoustic_dvae_checkpoint=DVAE_CHECKPOINT,
    content_dvae_checkpoint=CONTENT_DVAE_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
    model_args=model_args,
    audio=audio_config,
    content_dvae_config=contentDVAE_config,
    acoustic_dvae_config=acousticDVAE_config,
    batch_size=24,
    eval_batch_size=24,
    num_loader_workers=24,
    epochs=100,
    print_step=50,
    plot_step=500,
    log_model_step=100,
    save_step=5000,
    print_eval=False,
    save_n_checkpoints=2,
    save_checkpoints=True,
    run_name="genVC",
    optimizer="AdamW",
    output_path="exp/genVC_contentVec_LibriTTS",
    optimizer_wd_only_on_weights=True,
    lr=1e-4,
    optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-6},
    lr_scheduler="MultiStepLR",
    lr_scheduler_params={"milestones": [10, 25, 35, 50], "gamma": 0.5, "last_epoch": -1},
    weight_decay=1e-6,
    warmup_steps=4000,
    max_grad_norm=1.0,
    train_metafile='metafiles/libritts/train.txt',
    test_metafile='metafiles/libritts/test.txt',
    use_wandb=True,
    wandb_project='genVC',
    wandb_run_name='libritts',
)

if __name__ == '__main__':
    restore_path = None

    model = GPTTrainer.init_from_config(config)
    trainer_args = TrainerArgs(restore_path=restore_path)
    trainer = Trainer(
            trainer_args,
            config,
            model=model,
            output_path=config.output_path,
    )

    trainer.fit()
