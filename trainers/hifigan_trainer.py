from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torchaudio
from coqpit import Coqpit
from torch.nn import functional as F
from torch.utils.data import DataLoader
from trainer.torch import DistributedSampler
from trainer import TrainerModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from trainer.utils.distributed import get_rank
from layers.gpt import GPT
import random
from dataset import VCWaveDataset
from utils import TorchMelSpectrogram
from layers.dvae import DiscreteVAE
from layers.hifigan import HiFiGAN, MultiScaleDiscriminator, MultiPeriodDiscriminator, MultiScaleSTFTDiscriminator, MultiScaleSubbandCQTDiscriminator
from layers.content_processor import ContentvecExtractor
from layers.hifigan_loss import feature_criterion, discriminator_criterion, generator_criterion, mel_criterion
import wandb

class HiFiGANTrainer(TrainerModel):
    def __init__(self, config: Coqpit):

        super().__init__()
        self.config = config

        # init causal transformer model
        self.gpt = GPT(
            layers=self.config.model_args.gpt_layers,
            model_dim=self.config.model_args.gpt_n_model_channels,
            start_text_token=self.config.model_args.gpt_start_text_token,
            stop_text_token=self.config.model_args.gpt_stop_text_token,
            heads=self.config.model_args.gpt_n_heads,
            max_text_tokens=self.config.model_args.gpt_max_text_tokens,
            max_mel_tokens=self.config.model_args.gpt_max_audio_tokens,
            max_prompt_tokens=self.config.model_args.gpt_max_prompt_tokens,
            number_text_tokens=self.config.model_args.gpt_number_text_tokens,
            num_audio_tokens=self.config.model_args.gpt_num_audio_tokens,
            start_audio_token=self.config.model_args.gpt_start_audio_token,
            stop_audio_token=self.config.model_args.gpt_stop_audio_token,
            code_stride_len=self.config.model_args.gpt_code_stride_len,
        )

        self.hifigan = HiFiGAN(
            config.vocoder_config.input_feat_dim,
            config.vocoder_config.upsample_initial_channel,
            config.vocoder_config.resblock_kernel_sizes,
            config.vocoder_config.resblock_dilation_sizes,
            config.vocoder_config.upsample_rates,
            config.vocoder_config.upsample_kernal_sizes,
            config.vocoder_config.resblock_type,
        )
        self.hifigan_scale_factor = self.config.model_args.gpt_code_stride_len / self.config.vocoder_config.hop_length

        self.hifigan_discriminator = {
            "MSD_Discriminator": MultiScaleDiscriminator(),
            "MPD_Discriminator": MultiPeriodDiscriminator(
                config.vocoder_config.mpd_reshapes,
                config.vocoder_config.mpd_discriminator_channel_mult_factor,
                config.vocoder_config.mpd_use_spectral_norm),
            "MSTFT_Discriminator": MultiScaleSTFTDiscriminator(
                config.vocoder_config.msstftd_filters),
            "MSCQT_Discriminator": MultiScaleSubbandCQTDiscriminator(
                config.vocoder_config.mssbcqtd_filters,
                config.vocoder_config.mssbcqtd_max_filters,
                config.vocoder_config.mssbcqtd_filters_scale,
                config.vocoder_config.mssbcqtd_dilations,
                config.vocoder_config.mssbcqtd_in_channels,
                config.vocoder_config.mssbcqtd_out_channels,
                config.vocoder_config.sample_rate,
                config.vocoder_config.mssbcqtd_hop_lengths,
                config.vocoder_config.mssbcqtd_n_octavess,
                config.vocoder_config.mssbcqtd_bins_per_octave,)
        }

        # build criterion
        self.feature_criterion = feature_criterion()
        self.discriminator_criterion = discriminator_criterion()
        self.generator_criterion = generator_criterion()
        self.mel_criterion = mel_criterion(self.config.vocoder_config)

        self.content_extractor = ContentvecExtractor(config)
        self.eval_wav_samples = None

        if self.config.use_wandb and get_rank() == 0:
            wandb.init(
                project=config.wandb_project, 
                name=config.wandb_run_name)
            wandb.watch(self.hifigan)
            columns = ["Source", "Reconstructed Audio"]
            self.wandb_table = wandb.Table(columns=columns)

        # load GPT if available
        if self.config.model_args.gpt_checkpoint:
            self.load_checkpoint(self.gpt, "gpt", self.config.model_args.gpt_checkpoint)


        if self.config.model_args.hifigan_checkpoint:
            self.load_checkpoint(self.hifigan, "hifigan", self.config.model_args.hifigan_checkpoint)

        # Mel spectrogram extractor for conditioning
        self.torch_mel_spectrogram_style_encoder = TorchMelSpectrogram(
            filter_length=2048,
            hop_length=256,
            win_length=1024,
            normalize=False,
            sampling_rate=config.audio.sample_rate,
            mel_fmin=0,
            mel_fmax=8000,
            n_mel_channels=80,
            mel_norm_file=self.config.model_args.mel_norm_file,
        )

        # Load acoustic-DVAE
        assert self.config.acoustic_dvae_config.num_tokens == self.config.model_args.gpt_num_audio_tokens - 2
        self.acoustic_sample_rate = self.config.acoustic_dvae_config.audio.dvae_sample_rate
        self.acoustic_dvae = DiscreteVAE(
            channels=config.acoustic_dvae_config.num_channels,
            normalization=None,
            positional_dims=1,
            num_tokens=config.acoustic_dvae_config.num_tokens,
            codebook_dim=config.acoustic_dvae_config.codebook_dim,
            hidden_dim=config.acoustic_dvae_config.hidden_dim,
            num_resnet_blocks=config.acoustic_dvae_config.num_resnet_blocks,
            kernel_size=config.acoustic_dvae_config.kernel_size,
            num_layers=config.acoustic_dvae_config.num_layers,
            use_transposed_convs=False,
        )


        if self.config.acoustic_dvae_checkpoint:
            self.load_checkpoint(self.acoustic_dvae, "dvae", self.config.acoustic_dvae_checkpoint)
        elif not self.config.is_inference:
            raise RuntimeError(
                "You need to specify config.acoustic_dvae_checkpoint path to be able to train the GenVC!!"
            )

        # Mel spectrogram extractor for DVAE
        self.torch_mel_spectrogram_dvae = TorchMelSpectrogram(
            mel_norm_file=self.config.model_args.mel_norm_file, sampling_rate=self.acoustic_sample_rate
        )

        self.content_sample_rate = self.config.content_dvae_config.audio.dvae_sample_rate
        # Load Content-DVAE
        assert self.config.content_dvae_config.num_tokens == self.config.model_args.gpt_number_text_tokens - 2 
        self.content_dvae = DiscreteVAE(
            channels=config.content_dvae_config.num_channels,
            normalization=None,
            positional_dims=1,
            num_tokens=config.content_dvae_config.num_tokens,
            codebook_dim=config.content_dvae_config.codebook_dim,
            hidden_dim=config.content_dvae_config.hidden_dim,
            num_resnet_blocks=config.content_dvae_config.num_resnet_blocks,
            kernel_size=config.content_dvae_config.kernel_size,
            num_layers=config.content_dvae_config.num_layers,
            use_transposed_convs=False,
        )

        if self.config.content_dvae_checkpoint:
            self.load_checkpoint(self.content_dvae, "dvae", self.config.content_dvae_checkpoint)
        elif not self.config.is_inference:
            raise RuntimeError(
                "You need to specify config.content_dvae_checkpoint path to be able to train the GPT decoder!!"
            )

    def load_checkpoint(self, model, model_name, path):
        ckpt = torch.load(path, map_location=torch.device("cpu"))
        if "model" in ckpt.keys() and "config" in ckpt.keys():
            print("Coqui Trainer checkpoint detected! Converting it!")
            model_checkpoint = ckpt["model"]
            states_keys = list(model_checkpoint.keys())
            for key in states_keys:
                if model_name in key:
                    new_key = key.replace(model_name + ".", "", 1)
                    model_checkpoint[new_key] = model_checkpoint[key]
                    del model_checkpoint[key]
                else:
                    del model_checkpoint[key]
            model.load_state_dict(model_checkpoint, strict=True)
        else:
            model.load_state_dict(ckpt)
        print(f">> {model_name} weights restored from:", path)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        ...

    def optimize(self, batch, trainer):
        loss_dict = {}
        # total_loss = 0

        generator_losses = {}
        generator_total_loss = 0
        discriminator_losses = {}
        discriminator_total_loss = 0

        # Use input feature to get predictions
        mel_input = batch["mel_latents"]
        audio_gt = batch["wav"]

        mel_input = torch.nn.functional.interpolate(
            mel_input.transpose(1, 2),
            scale_factor=[self.hifigan_scale_factor],
            mode="linear",
        ).squeeze(1)
        audio_pred = self.hifigan.forward(mel_input)

        # Calculate and BP Discriminator losses
        trainer.optimizer[0].zero_grad()

        for key, _ in self.hifigan_discriminator.items():
            y_r, y_g, _, _ = self.hifigan_discriminator[key].forward(
                audio_gt, audio_pred.detach()
            )
            (
                discriminator_losses["{}_loss".format(key)],
                _,
                _,
            ) = self.discriminator_criterion(y_r, y_g)
            discriminator_total_loss += discriminator_losses[
                "{}_loss".format(key)
            ]
        
        discriminator_total_loss.backward()

        trainer.optimizer[0].step()

        # Calculate and BP Generator losses
        trainer.optimizer[1].zero_grad()
        for key, _ in self.hifigan_discriminator.items():
            y_r, y_g, f_r, f_g = self.hifigan_discriminator[key].forward(
                audio_gt, audio_pred
            )
            # print(f_r[0].shape, f_g[0].shape)
            generator_losses["{}_featureLoss".format(key)] = self.feature_criterion(
                f_r, f_g
            )
            generator_losses["{}_generatorLoss".format(key)], _ = self.generator_criterion(y_g)
            generator_total_loss += generator_losses["{}_featureLoss".format(key)]
            generator_total_loss += generator_losses["{}_generatorLoss".format(key)]

        generator_losses["mel"] = self.mel_criterion(audio_gt, audio_pred)
        generator_total_loss += generator_losses["mel"]

        generator_total_loss.backward()
        # _, _ = self.scaled_backward(generator_total_loss, None, trainer, self.generator_optimizer)
        trainer.optimizer[1].step()

        # Get the total losses
        # total_loss = discriminator_total_loss + generator_total_loss
        loss_dict["mel_loss"] = generator_losses["mel"].item()
        loss_dict["loss_gen"] = generator_total_loss.item()
        loss_dict["loss_disc"] = discriminator_total_loss.item()
        self.loss_dict = loss_dict

        return {"model_outputs": None}, loss_dict

    def format_batch(self, batch: Dict) -> Dict:
        return batch

    @torch.no_grad()  # torch no grad to avoid gradients from the pre-processing and DVAE codes extraction
    def format_batch_on_device(self, batch):
        """Compute spectrograms on the device."""
        # compute conditioning mel specs
        # transform waves from torch.Size([B, num_cond_samples, 1, T] to torch.Size([B * num_cond_samples, 1, T] because if is faster than iterate the tensor
        B, num_cond_samples, C, T = batch["conditioning"].size()
        conditioning_reshaped = batch["conditioning"].view(B * num_cond_samples, C, T)
        paired_conditioning_mel = self.torch_mel_spectrogram_style_encoder(conditioning_reshaped)
        n_mel = self.torch_mel_spectrogram_style_encoder.n_mel_channels  
        T_mel = paired_conditioning_mel.size(2)
        paired_conditioning_mel = paired_conditioning_mel.view(B, num_cond_samples, n_mel, T_mel)
        # get the conditioning embeddings
        batch["cond_mels"] = paired_conditioning_mel
        # compute codes using DVAE
        if self.config.audio.sample_rate != self.acoustic_sample_rate:
            acoustic_dvae_wav = torchaudio.functional.resample(
                batch["wav"],
                orig_freq=self.config.audio.sample_rate,
                new_freq=self.acoustic_sample_rate,
                # remove the following parameters if you want to use the default values
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="kaiser_window",
                beta=14.769656459379492,
            )
        else:
            acoustic_dvae_wav = batch["wav"]
        
        dvae_mel_spec = self.torch_mel_spectrogram_dvae(acoustic_dvae_wav)
        audio_codes = self.acoustic_dvae.get_codebook_indices(dvae_mel_spec)

        batch["audio_codes"] = audio_codes
        _, audio_code_len = audio_codes.shape

        # compute phonetic codes
        if self.config.audio.sample_rate != self.content_sample_rate:
            content_wav = torchaudio.functional.resample(
                batch["wav"],
                orig_freq=self.config.audio.sample_rate,
                new_freq=self.content_sample_rate
            )
        else:
            content_wav = batch["wav"]

        content_wav = F.pad(content_wav, (0, int(self.config.text_frame_rate * self.content_sample_rate)))
        content_feat = self.content_extractor.extract_content_features(content_wav.squeeze())
        content_feat = content_feat.transpose(1, 2)
        text_codes = self.content_dvae.get_codebook_indices(content_feat)

        batch["text_inputs"] = text_codes
        batch["text_lengths"] = batch["text_lengths"].to(torch.long)
        batch["wav_lengths"] = batch["wav_lengths"] + (self.config.model_args.gpt_code_stride_len // 2) 
        
        # compute mel latents
        batch["mel_latents"] = self.gpt(
            batch["text_inputs"],
            batch["text_lengths"],
            batch["audio_codes"],
            batch["wav_lengths"],
            cond_mels=batch["cond_mels"],
            cond_lens=batch["cond_lens"],
            return_latent=True,
        )

        # padding to the same length
        wav_expected_len = int(audio_code_len * self.config.model_args.gpt_code_stride_len)
        batch["wav"] = F.pad(batch["wav"], pad=(0, wav_expected_len - batch["wav"].shape[-1]), mode="constant", value=0)
        batch["wav"] = batch["wav"][:,:,:wav_expected_len]
        
        # delete useless batch tensors
        del batch["cond_mels"]
        del batch["cond_lens"]
        del batch["audio_codes"]
        del batch["text_lengths"]
        del batch["text_inputs"]
        del batch["conditioning"]

        return batch

    @torch.no_grad()
    def eval_step(self, batch, criterion):
        # ignore masking for more consistent evaluation
        # batch["cond_idxs"] = None
        mel_input = batch["mel_latents"]

        mel_input = torch.nn.functional.interpolate(
            mel_input.transpose(1, 2),
            scale_factor=[self.hifigan_scale_factor],
            mode="linear",
        ).squeeze(1)

        discriminator_total_loss = 0
        audio_gt = batch["wav"]
        audio_pred = self.hifigan.forward(mel_input)
        discriminator_losses = {}
        for key, _ in self.hifigan_discriminator.items():
            y_r, y_g, _, _ = self.hifigan_discriminator[key].forward(
                audio_gt, audio_pred.detach()
            )
            (
                discriminator_losses["{}_loss".format(key)],
                _,
                _,
            ) = self.discriminator_criterion(y_r, y_g)
            discriminator_total_loss += discriminator_losses[
                "{}_loss".format(key)
            ]
        mel_loss = self.mel_criterion(audio_gt, audio_pred)

        if self.eval_plot and self.config.use_wandb and get_rank() == 0:
            idx = random.randint(0, audio_gt.size(0) - 1)
            wav_gt_sample = batch["wav"][idx].squeeze().cpu().numpy()
            wav_pred_sample = audio_pred[idx].squeeze().cpu().numpy()
            wav_gt = wandb.Audio(wav_gt_sample, sample_rate=self.config.audio.sample_rate)
            wav_pred = wandb.Audio(wav_pred_sample, sample_rate=self.config.audio.sample_rate)    
            self.eval_wav_samples = (wav_gt, wav_pred)
            self.eval_plot = False

        return {"model_outputs": None}, {"loss_disc": discriminator_total_loss.item(), "mel_loss": mel_loss.item()}

    def on_train_epoch_start(self, trainer):
        self.acoustic_dvae.eval()
        self.acoustic_dvae.to(self.device)
        self.content_dvae.eval()
        self.content_dvae.to(self.device)
        self.content_extractor.model.eval()
        self.content_extractor.model.to(self.device)
        self.gpt.eval()
        self.gpt.to(self.device)
        self.hifigan.train()
        self.hifigan.to(self.device)
        self.eval_plot = True
        for key, _ in self.hifigan_discriminator.items():
            self.hifigan_discriminator[key].train()
            self.hifigan_discriminator[key].to(self.device)

    def on_train_step_end(self, trainer):
        if self.config.use_wandb and get_rank() == 0:
            log_dict = {'step': trainer.total_steps_done,
                        'current_lr': trainer.optimizer[0].param_groups[0]["lr"]}
            for k, v in self.loss_dict.items():
                if 'loss' in k:
                    log_dict[k] = v

            wandb.log(log_dict)

    def on_epoch_end(self, trainer):
        if self.config.use_wandb and get_rank() == 0:
            log_dict = {'epoch': trainer.epochs_done}
            for k, v in trainer.keep_avg_train.avg_values.items():
                if 'loss' in k:
                    log_dict[k] = v

            for k, v in trainer.keep_avg_eval.avg_values.items():
                if 'loss' in k:
                    log_dict['Eval_' + k] = v
            wandb.log(log_dict)

            if self.eval_wav_samples:
                wav_gt, wav_pred = self.eval_wav_samples
                # the table on wandb will only show the last synthesized spectrograms
                self.wandb_table.add_data(wav_gt, wav_pred)
                wandb.log({self.config.wandb_run_name + " Wav Samples": self.wandb_table})
                # to show all the synthesized spectrograms, comment the above lines and uncomment the following lines, however, it will take a lot of space on wandb
                # _table = wandb.Table(columns=self.wandb_table.columns, data=self.wandb_table.data)
                # _table.add_data(wav_gt, wav_pred)
                # wandb.log({self.config.wandb_run_name + " Wav Samples": _table})
                # self.wandb_table = _table
                self.eval_wav_samples = None
                self.wandb_table = wandb.Table(columns=self.wandb_table.columns)

    @torch.inference_mode()
    def get_gpt_cond_latents(self, audio, sr, length: int = 30, chunk_length: int = 6):
        style_embs = []
        if audio.shape[1] > sr * length:
            audio = audio[:, : sr * length]
        for i in range(0, audio.shape[1], sr * chunk_length):
            audio_chunk = audio[:, i : i + sr * chunk_length]

            # if the chunk is too short ignore it 
            if audio_chunk.size(-1) < sr * 0.33:
                continue

            mel_chunk = self.torch_mel_spectrogram_style_encoder(audio_chunk.unsqueeze(0))
            style_emb = self.gpt.get_style_emb(mel_chunk.to(self.device), None)
            style_embs.append(style_emb)

        cond_latent = torch.stack(style_embs).mean(dim=0)
        return cond_latent.transpose(1, 2)
    
    @torch.no_grad()
    def inference(
        self,
        src_audio: torch.Tensor,
        cond_latent: torch.Tensor,
        do_sample: bool = True,
        top_p: float = 0.85,
        top_k: int = 15,
        temperature: float = 0.75,
        num_beams: int = 1,
        length_penalty: float = 1.0,
        repetition_penalty: float = 10.0,
        output_attentions: bool = False,
    ):  # pylint: disable=dangerous-default-value
        
        content_feat = self.content_extractor.extract_content_features(src_audio)
        content_codes = self.content_dvae.get_codebook_indices(content_feat.transpose(1, 2))
        gen_codes = self.gpt.generate(
            cond_latent,
            content_codes,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_beams=num_beams,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            output_attentions=output_attentions,
        )[0]
        gen_codes = gen_codes[(gen_codes!=self.gpt.stop_audio_token).nonzero().squeeze()]
        expected_output_len = torch.tensor([gen_codes.shape[-1] * self.config.model_args.gpt_code_stride_len], device=self.device)
        content_len = torch.tensor([content_codes.shape[-1]], device=self.device)
        acoustic_latents = self.gpt(content_codes,
                                    content_len,
                                    gen_codes.unsqueeze(0),
                                    expected_output_len,
                                    cond_latents=cond_latent,
                                    return_latent=True)
        mel_input = torch.nn.functional.interpolate(
            acoustic_latents.transpose(1, 2),
            scale_factor=[self.hifigan_scale_factor],
            mode="linear",
        ).squeeze(1)
        audio_pred = self.hifigan.forward(mel_input)
        return audio_pred

    @staticmethod
    def get_criterion():
        return None

    def get_sampler(self, dataset: VCWaveDataset, num_gpus=1):
        # sampler for DDP
        batch_sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        return batch_sampler

    def get_data_loader(
        self,
        config: Coqpit,
        assets: Dict,
        is_eval: bool,
        samples: Union[List[Dict], List[List]],
        verbose: bool,
        num_gpus: int,
        rank: int = None,
    ) -> "DataLoader":  # pylint: disable=W0613
        # init dataloader
        dataset = VCWaveDataset(config.model_args, config.test_metafile if is_eval else config.train_metafile, self.config.audio.sample_rate, config.text_frame_rate)

        # wait all the DDP process to be ready
        if num_gpus > 1:
            torch.distributed.barrier()
        # get samplers
        sampler = self.get_sampler(dataset, num_gpus)

        if sampler is None:
            loader = DataLoader(
                dataset,
                batch_size = config.eval_batch_size if is_eval else config.batch_size,
                collate_fn=dataset.collate_fn,
                num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                pin_memory=False,
                drop_last=True,
            )
        else:
            loader = DataLoader(
                dataset,
                sampler=sampler,
                batch_size = config.eval_batch_size if is_eval else config.batch_size,
                collate_fn=dataset.collate_fn,
                num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                pin_memory=False,
                drop_last=True,
            )
        return loader

    def get_optimizer(self) -> List:
        """Initiate and return the optimizer based on the config parameters."""
        optimizer_params_discriminator = []
        for discriminator in self.hifigan_discriminator.keys():
            optimizer_params_discriminator.append(
                dict(params=self.hifigan_discriminator[discriminator].parameters())
            )
        discriminator_optimizer = AdamW(
            optimizer_params_discriminator,
            lr=self.config.lr,
            betas=self.config.optimizer_params["betas"],
        )

        optimizer_params_generator = [dict(params=self.hifigan.parameters())]
        generator_optimizer = AdamW(
            optimizer_params_generator,
            lr=self.config.lr,
            betas=self.config.optimizer_params["betas"],
        )

        return [discriminator_optimizer, generator_optimizer]

    def get_scheduler(self, optimizer) -> List:
        """Set the scheduler for the optimizer.

        Args:
            optimizer: `torch.optim.Optimizer`.
        """
        discriminator_scheduler = ExponentialLR(
            optimizer[0],
            gamma=self.config.lr_decay,
            last_epoch=-1,
        )

        generator_scheduler = ExponentialLR(
            optimizer[1],
            gamma=self.config.lr_decay,
            last_epoch=-1,
        )
        return [discriminator_scheduler, generator_scheduler] 


    @staticmethod
    def init_from_config(config):
        """Initiate model from config
        Args:
            config (GPTTrainerConfig): Model config.
        """
        return HiFiGANTrainer(config)