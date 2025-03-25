from typing import Dict, List, Union

import torch
from coqpit import Coqpit
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from trainer.torch import DistributedSampler
from trainer import TrainerModel
from torch.optim import Adam
from trainer.utils.distributed import get_rank
from utils import TorchMelSpectrogram, plot_feat
from layers.dvae import DiscreteVAE
from dataset import Waveform_DVAEDataset
from layers.content_processor import ContentvecExtractor, MultiLingualContentExtractor
import wandb

class VAE_Trainer(TrainerModel):
    def __init__(self, config: Coqpit):
        
        super().__init__()
        self.config = config

        self.dvae = DiscreteVAE(
            channels=config.num_channels,
            normalization=None,
            positional_dims=1,
            num_tokens=config.num_tokens,
            codebook_dim=config.codebook_dim,
            hidden_dim=config.hidden_dim,
            num_resnet_blocks=config.num_resnet_blocks,
            kernel_size=config.kernel_size,
            num_layers=config.num_layers,
            use_transposed_convs=False,
        )

        if self.config.use_wandb and get_rank() == 0:
            wandb.init(project=config.wandb_project, name=config.wandb_run_name)
            wandb.watch(self.dvae)
            columns = ["Epoch", "Input Feature", "Reconstructed Feature"]
            self.wandb_table = wandb.Table(columns=columns)
        
        self.eval_samples = None
        if self.config.vae_checkpoint:
            self.load_checkpoint(self.config.vae_checkpoint)

        if self.config.feat_type == 'Mel-spectrogram':
            self.feat_extractor = TorchMelSpectrogram(mel_norm_file=config.mel_norm_file, sampling_rate=config.audio.sample_rate)
        elif self.config.feat_type == 'ContentVec':
            self.feat_extractor = ContentvecExtractor(config)
        elif self.config.feat_type == 'W2V2_BERT':
            self.feat_extractor = MultiLingualContentExtractor()
        else:
            raise ValueError(f"Unknown feature type {self.config.feat_type}")

    @property
    def device(self):
        return next(self.parameters()).device
    
    def format_batch(self, batch: Dict) -> Dict:
        return batch

    def forward(self, x):
        ...

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=torch.device("cpu"))
        if "model" in ckpt.keys() and "config" in ckpt.keys():
            print("Coqui Trainer checkpoint detected! Converting it!")
            dvae_checkpoint = ckpt["model"]
            states_keys = list(dvae_checkpoint.keys())
            for key in states_keys:
                if "dvae" in key:
                    new_key = key.replace("dvae.", "", 1)
                    dvae_checkpoint[new_key] = dvae_checkpoint[key]
                    del dvae_checkpoint[key]
            self.dvae.load_state_dict(dvae_checkpoint)
        else:
            self.dvae.load_state_dict(ckpt)
        print(f"Loaded VAE checkpoint from {path}")
            
    @torch.no_grad() 
    def format_batch_on_device(self, batch):
        """Compute spectrograms on the device."""
        if self.config.feat_type == 'Mel-spectrogram':
            input_wav = batch['wav']
            batch['feat'] = self.feat_extractor(input_wav)
        else:
            input_wav = batch['wav'].squeeze()
            batch['feat'] = self.feat_extractor.extract_content_features(input_wav).transpose(1, 2)

        # the default stride of the convs in the DVAE is 2 and the number of layers is 2, so the input should be divisible by 4 for training
        remainder = batch['feat'].shape[-1] % 4
        if remainder:
            batch['feat'] = batch['feat'][:, :, :-remainder]
        
        return batch

    def optimize(self, batch, trainer):
        loss_dict = {}

        # to-do: add mask for loss calculation
        recon_loss, commitment_loss, out = self.dvae(batch['feat'])
        commitment_loss = commitment_loss.mean()
        total_loss = recon_loss + commitment_loss

        total_loss.backward()
        clip_grad_norm_(self.dvae.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

        loss_dict['recon_loss'] = recon_loss
        loss_dict['commitment_loss'] = commitment_loss
        loss_dict['loss'] = total_loss
        self.loss_dict = loss_dict

        return {"model_outputs": None}, loss_dict
    
    @torch.no_grad()
    def eval_step(self, batch, trainer):
        loss_dict = {}
        recon_loss, commitment_loss, out = self.dvae(batch['feat'])
        commitment_loss = commitment_loss.mean()
        total_loss = recon_loss + commitment_loss
        loss_dict['recon_loss'] = recon_loss.item()
        loss_dict['commitment_loss'] = commitment_loss.item()
        loss_dict['loss'] = total_loss.item()

        update_eval_values = {}
        for k, v in loss_dict.items():
            update_eval_values["avg_" + k] = v
        
        trainer.keep_avg_eval.update_values(update_eval_values)
        # plot randomly selected spectrograms
        if self.config.use_wandb and get_rank() == 0 and self.eval_plot:
            index = torch.randint(0, out.size(0), (1,)).item()
            feat_input = batch['feat'][index]
            feat_output = out[index]
            spec_input = wandb.Image(plot_feat(feat_input.detach().cpu()))
            spec_output = wandb.Image(plot_feat(feat_output.detach().cpu()))
            self.eval_samples = [spec_input, spec_output]
            self.eval_plot = False

        return {"model_outputs": None}, loss_dict
    
    def on_train_epoch_start(self, trainer):
        self.dvae.train()
        self.dvae.to(self.device)
        if self.config.feat_type != 'Mel-spectrogram':
            self.feat_extractor.model.to(self.device)
            self.feat_extractor.model.eval()
        self.eval_plot = True

    def on_train_step_end(self, trainer):
        if self.config.use_wandb and get_rank() == 0:
            log_dict = {'step': trainer.total_steps_done,
                        'recon_loss': self.loss_dict['recon_loss'].item(),
                        'commitment_loss': self.loss_dict['commitment_loss'].item(),
                        'loss': self.loss_dict['loss'].item(),
                        'current_lr': trainer.optimizer.param_groups[0]["lr"]}
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

            if self.eval_samples:
                spec_input, spec_output = self.eval_samples
                # the table on wandb will only show the last synthesized spectrograms
                self.wandb_table.add_data(trainer.epochs_done, spec_input, spec_output)
                wandb.log({self.config.wandb_run_name + " Features": self.wandb_table})
                # to show all the synthesized spectrograms, comment the above lines and uncomment the following lines, however, it will take a lot of space on wandb
                # _table = wandb.Table(columns=self.wandb_table.columns, data=self.wandb_table.data)
                # _table.add_data(trainer.epochs_done, spec_input, spec_output)
                # wandb.log({self.config.wandb_run_name + " Features": _table})
                # self.wandb_table = _table
                self.eval_samples = None

    @torch.no_grad()
    def inference(
        self,
        x,
        aux_input=None,
    ):  
        return None
    
    @staticmethod
    def get_criterion():
        return None
    
    def get_sampler(self, dataset: Waveform_DVAEDataset, num_gpus=1):
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

        dataset = Waveform_DVAEDataset(config.test_metafile if is_eval else config.train_metafile, is_eval, self.config.audio.dvae_sample_rate, config.max_wav_len)

        # wait all the DDP process to be ready
        if num_gpus > 1:
            torch.distributed.barrier()
        # get samplers
        sampler = self.get_sampler(dataset, num_gpus)
        if sampler is None:
            loader = DataLoader(
                dataset,
                shuffle=False if is_eval else True,
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
        """Initiate and return the optimizer"""
        self.optimizer = Adam(self.dvae.parameters(), lr=self.config.lr, betas=self.config.opt_betas)
        return self.optimizer
    
    @staticmethod
    def init_from_config(config: "VAETrainerConfig", samples: Union[List[List], List[Dict]] = None):
        return VAE_Trainer(config)