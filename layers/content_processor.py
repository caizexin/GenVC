import os
import torch
from fairseq import checkpoint_utils
from transformers import HubertModel

# use python < 3.11 https://github.com/facebookresearch/fairseq/issues/5012
class ContentvecExtractor(torch.nn.Module):
    def __init__(self, cfg):
        super(ContentvecExtractor, self).__init__()
        self.model_path = cfg.contentvec_model_path
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [self.model_path]
        )
        self.model = models[0]
        self.model.eval()

    def extract_content_features(self, wavs):
        """extract content features from a batch of dataloader
        Args:
            wavs: tensor (batch, T)
        """
        device = next(self.model.parameters()).device
        wavs = wavs.to(device)  # (batch, max_len)
        padding_mask = torch.eq(wavs, torch.zeros_like(wavs)).to(device)
        with torch.no_grad():
            logits = self.model.extract_features(
                source=wavs, padding_mask=padding_mask, output_layer=12
            )
            # feats: (batch, T, 256)
            feats = self.model.final_proj(logits[0])
        return feats
    
    def forward(self, wavs):
        return self.extract_content_features(wavs)

class MultiLingualContentExtractor():
    def __init__(self):
        self.model = HubertModel.from_pretrained("utter-project/mHuBERT-147")
        self.model.eval()

    def extract_content_features(self, wavs):
        """extract features from a batch of dataloader
        Args:
            wavs: tensor (batch, T)
        """
        device = next(self.model.parameters()).device
        wavs = wavs.to(device)  # (batch, max_len)
        with torch.no_grad():
            feats = self.model.feature_extractor(wavs)
            # feats: (batch, T, 512)
            feats = feats.transpose(1, 2)
        return feats