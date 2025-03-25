from configs.vae_config import VAEConfig
from configs.base_configs import BaseAudioConfig
from trainers.VAE_trainer import VAE_Trainer
from trainer import Trainer, TrainerArgs

MEL_NORM_FILE = 'pre_trained/mel_stats.pth'
audio_config = BaseAudioConfig(dvae_sample_rate=16000)

vae_config = VAEConfig(
    audio=audio_config,
    mel_norm_file=MEL_NORM_FILE,
    feat_type='ContentVec',
    contentvec_model_path='pre_trained/contentVec.pt',
    warmup_steps=1000,
    batch_size=64,
    eval_batch_size=16,
    save_n_checkpoints=2,
    save_step=2000,
    lr=1e-4,
    num_loader_workers=24,
    num_eval_loader_workers=16,
    max_wav_len=audio_config.dvae_sample_rate * 6,
    train_metafile='metafiles/libritts/train.txt',
    test_metafile='metafiles/libritts/test.txt',
    epochs=200,
    grad_clip_norm=0.5,
    output_path='exp/content_dvae',
    use_wandb=True,
    vae_checkpoint=None,
    wandb_project='content_dvae',
    wandb_run_name='libritts',
    num_channels=256,
    num_tokens=256,
    codebook_dim=512,
    hidden_dim=512,
    num_resnet_blocks=3,
    kernel_size=3,
    num_layers=2,
)

if __name__ == '__main__':
    model = VAE_Trainer.init_from_config(vae_config)
    restore_path = None
    trainer_args = TrainerArgs(restore_path=restore_path)
    trainer = Trainer(trainer_args,
                    vae_config,
                    model=model,
                    output_path=vae_config.save_dir)

    trainer.fit()
