# ported and adapted from: https://github.com/coqui-ai/TTS
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torchaudio
from coqpit import Coqpit
from torch.nn import functional as F
from torch.utils.data import DataLoader
from trainer.torch import DistributedSampler
from trainer import TrainerModel
from trainer.trainer_utils import get_optimizer, get_scheduler
from layers.gpt import GPT
from dataset import VCWaveDataset
from utils import TorchMelSpectrogram
from trainer.utils.distributed import get_rank
from layers.dvae import DiscreteVAE
from layers.content_processor import ContentvecExtractor
import wandb

class GPTTrainer(TrainerModel):
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

        if self.config.use_wandb and get_rank() == 0:
            wandb.init(
                project=config.wandb_project, 
                name=config.wandb_run_name)
            wandb.watch(self.gpt)
            columns = ["Source", "Generated Utt", "Target Voice", "Converted Utt"]
            self.wandb_table = wandb.Table(columns=columns)

        self.content_extractor = ContentvecExtractor(config)
        self.eval_wav_samples = None

        # load GPT if available
        if self.config.model_args.gpt_checkpoint:
            self.load_checkpoint(self.gpt, "gpt", self.config.model_args.gpt_checkpoint)

        # Mel spectrogram extractor for conditioning
        # The following mel spectorgram extractor is the one we used in our paper, but we recommend to use self.torch_mel_spectrogram_dvae if you train a new model from scratch, no need to have two extractors
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
        else:
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
        else:
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

    def generate_eval_sample(self, batch):
        """
        Generate samples on evaluation set for listening
        """
        text_inputs = batch["text_inputs"]
        cond_mels = batch["cond_mels"]
        cond_lens = batch["cond_lens"]
        text_lengths = batch["text_lengths"]
        src_id, src_gen_mel_code, tgt_id, cv_mel_code = self.gpt.eval_sample(text_inputs, text_lengths, cond_mels=cond_mels, cond_lens=cond_lens)
        wav_sample = batch['wav'][src_id]
        wav_sample_len = batch['wav_lengths'][src_id]
        tgt_wav_sample = batch['conditioning'][tgt_id].squeeze(0)

        src_gen_mel, _ = self.acoustic_dvae.decode(src_gen_mel_code.unsqueeze(0))
        src_gen_wav = self.torch_mel_spectrogram_dvae.invert(src_gen_mel)

        cv_mel, _ = self.acoustic_dvae.decode(cv_mel_code.unsqueeze(0))
        cv_wav = self.torch_mel_spectrogram_dvae.invert(cv_mel)

        wav_org = wandb.Audio(wav_sample.squeeze().cpu().numpy()[:wav_sample_len], sample_rate=self.config.audio.sample_rate)
        wav_tgt = wandb.Audio(tgt_wav_sample.squeeze().cpu().numpy(), sample_rate=self.config.audio.sample_rate)
        wav_gen = wandb.Audio(src_gen_wav.squeeze(), sample_rate=self.config.audio.sample_rate)
        wav_cv = wandb.Audio(cv_wav.squeeze(), sample_rate=self.config.audio.sample_rate)

        self.eval_wav_samples = [wav_org, wav_gen, wav_tgt, wav_cv]


    def forward(self, text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels, cond_lens):
        """
        Forward pass that uses both content codes and voice 

        text_inputs: long tensor, (b,t)
        text_lengths: long tensor, (b,)
        mel_inputs:  long tensor, (b,m)
        wav_lengths: long tensor, (b,)
        cond_mels: MEL float tensor, (b, num_samples, 80,t_m)
        cond_lens: long tensor, (b,)
        """
        losses = self.gpt(
            text_inputs,
            text_lengths,
            audio_codes,
            wav_lengths,
            cond_mels=cond_mels,
            cond_lens=cond_lens,
        )

        return losses

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
        # compute phonetic code lengths
        batch["text_lengths"] = batch["text_lengths"].to(torch.long)

        # delete useless batch tensors
        # del batch["wav"]
        # del batch["conditioning"]
        del content_wav
        del content_feat

        return batch

    def train_step(self, batch, criterion):
        loss_dict = {}
        cond_mels = batch["cond_mels"]
        text_inputs = batch["text_inputs"]
        text_lengths = batch["text_lengths"]
        audio_codes = batch["audio_codes"]
        wav_lengths = batch["wav_lengths"]
        cond_lens = batch["cond_lens"]
        # cond_lens = batch["cond_lens"]

        loss_text, loss_mel, Top10Accuracy, _ = self.forward(
            text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels, cond_lens
        )

        loss_dict["loss_text_ce"] = loss_text
        loss_dict["loss_mel_ce"] = loss_mel
        loss_dict["loss"] = self.config.model_args.gpt_loss_text_ce_weight * loss_dict["loss_text_ce"] + self.config.model_args.gpt_loss_mel_ce_weight * loss_dict["loss_mel_ce"]
        loss_dict["top10acc"] = Top10Accuracy
        self.loss_dict = {k: v.item() for k, v in loss_dict.items()}

        return {"model_outputs": None}, loss_dict

    def eval_step(self, batch, criterion):
        if self.eval_plot and self.config.use_wandb and get_rank() == 0:
            self.generate_eval_sample(batch)
            self.eval_plot = False

        return self.train_step(batch, criterion)

    def on_train_epoch_start(self, trainer):
        self.acoustic_dvae.eval()
        self.acoustic_dvae.to(self.device)
        self.content_dvae.eval()
        self.content_dvae.to(self.device)
        self.content_extractor.model.eval()
        self.content_extractor.model.to(self.device)
        self.gpt.train()
        self.gpt.to(self.device)
        self.eval_plot = True # if trainer.epochs_done > 0 else False

    def on_train_step_end(self, trainer):
        if self.config.use_wandb and get_rank() == 0:
            log_dict = {'step': trainer.total_steps_done,
                        'current_lr': trainer.optimizer.param_groups[0]["lr"]}
            for k, v in self.loss_dict.items():
                if 'loss' in k or 'top10acc' in k:
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

            # only log audio samples if the loss is low enough
            # it should be lower than 4.0 if the number of acoustic tokens is 1024
            if self.eval_wav_samples and self.loss_dict["loss_mel_ce"] < 4.0:
                wav_org, wav_gen, wav_tgt, wav_cv = self.eval_wav_samples
                # the table on wandb will only show the last synthesized spectrograms
                self.wandb_table.add_data(wav_org, wav_gen, wav_tgt, wav_cv)
                wandb.log({self.config.wandb_run_name + " Audio Samples": self.wandb_table})
                # to show all the synthesized spectrograms, comment the above lines and uncomment the following lines, however, it will take a lot of space on wandb
                # _table = wandb.Table(columns=self.wandb_table.columns, data=self.wandb_table.data)
                # _table.add_data(wav_org, wav_gen, wav_tgt, wav_cv)
                # wandb.log({self.config.wandb_run_name + " Audio Samples": _table})
                # self.wandb_table = _table
                self.eval_wav_samples = None

    @torch.no_grad()
    def inference(
        self,
        x,
        aux_input=None,
    ):  # pylint: disable=dangerous-default-value
        return None

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
        # ToDo: deal with multi GPU training
        if self.config.optimizer_wd_only_on_weights:
            # parameters to only GPT model
            net = self.gpt

            # normalizations
            norm_modules = (
                nn.BatchNorm2d,
                nn.InstanceNorm2d,
                nn.BatchNorm1d,
                nn.InstanceNorm1d,
                nn.BatchNorm3d,
                nn.InstanceNorm3d,
                nn.GroupNorm,
                nn.LayerNorm,
            )
            # nn.Embedding
            emb_modules = (nn.Embedding, nn.EmbeddingBag)

            param_names_notweights = set()
            all_param_names = set()
            param_map = {}
            for mn, m in net.named_modules():
                for k, v in m.named_parameters():
                    v.is_bias = k.endswith(".bias")
                    v.is_weight = k.endswith(".weight")
                    v.is_norm = isinstance(m, norm_modules)
                    v.is_emb = isinstance(m, emb_modules)

                    fpn = "%s.%s" % (mn, k) if mn else k  # full param name
                    all_param_names.add(fpn)
                    param_map[fpn] = v
                    if v.is_bias or v.is_norm or v.is_emb:
                        param_names_notweights.add(fpn)

            params_names_notweights = sorted(list(param_names_notweights))
            params_notweights = [param_map[k] for k in params_names_notweights]
            params_names_weights = sorted(list(all_param_names ^ param_names_notweights))
            params_weights = [param_map[k] for k in params_names_weights]

            groups = [
                {"params": params_weights, "weight_decay": self.config.optimizer_params["weight_decay"]},
                {"params": params_notweights, "weight_decay": 0},
            ]
            # torch.optim.AdamW
            opt = get_optimizer(
                self.config.optimizer,
                self.config.optimizer_params,
                self.config.lr,
                parameters=groups,
            )
            opt._group_names = [params_names_weights, params_names_notweights]
            return opt

        return get_optimizer(
            self.config.optimizer,
            self.config.optimizer_params,
            self.config.lr,
            # optimize only for the GPT model
            parameters=self.gpt.parameters(),
        )

    def get_scheduler(self, optimizer) -> List:
        """Set the scheduler for the optimizer.
        Args:
            optimizer: `torch.optim.Optimizer`.
        """
        return get_scheduler(self.config.lr_scheduler, self.config.lr_scheduler_params, optimizer)

    @staticmethod
    def init_from_config(config):
        """Initiate model from config
        Args:
            config (GPTTrainerConfig): Model config.
        """
        return GPTTrainer(config)
