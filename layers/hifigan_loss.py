import torch
from librosa.filters import mel as librosa_mel_fn

mel_basis = {}
hann_window = {}    

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    # Min value: ln(1e-5) = -11.5129
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def extract_mel_features(
    y,
    cfg,
    center=False,
):
    """Extract mel features

    Args:
        y (tensor): audio data in tensor
        cfg (dict): configuration in cfg.preprocess
        center (bool, optional): In STFT, whether t-th frame is centered at time t*hop_length. Defaults to False.

    Returns:
        tensor: a tensor containing the mel feature calculated based on STFT result
    """
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    if cfg.mel_fmax not in mel_basis:
        mel = librosa_mel_fn(
            sr=cfg.sample_rate,
            n_fft=cfg.fft_size,
            n_mels=cfg.num_mels,
            fmin=cfg.mel_fmin,
            fmax=cfg.mel_fmax,
        )
        mel_basis[str(cfg.mel_fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(cfg.win_length).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((cfg.fft_size - cfg.hop_length) / 2), int((cfg.fft_size - cfg.hop_length) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    spec = torch.stft(
        y,
        cfg.fft_size,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(cfg.mel_fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)
    return spec.squeeze(0)


class feature_criterion(torch.nn.Module):
    def __init__(self):
        super(feature_criterion, self).__init__()

    def __call__(self, fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        loss = loss * 2

        return loss
    
class discriminator_criterion(torch.nn.Module):
    def __init__(self):
        super(discriminator_criterion, self).__init__()

    def __call__(self, disc_real_outputs, disc_generated_outputs):
        loss = 0
        r_losses = []
        g_losses = []

        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean((1 - dr) ** 2)
            g_loss = torch.mean(dg**2)
            loss += r_loss + g_loss
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())

        return loss, r_losses, g_losses
    
class generator_criterion(torch.nn.Module):
    def __init__(self):
        super(generator_criterion, self).__init__()

    def __call__(self, disc_outputs):
        loss = 0
        gen_losses = []

        for dg in disc_outputs:
            l = torch.mean((1 - dg) ** 2)
            gen_losses.append(l)
            loss += l

        return loss, gen_losses
    
class mel_criterion(torch.nn.Module):
    def __init__(self, cfg):
        super(mel_criterion, self).__init__()
        self.cfg = cfg
        self.l1Loss = torch.nn.L1Loss(reduction="mean")

    def __call__(self, y_gt, y_pred):
        loss = 0

        y_gt_mel = extract_mel_features(y_gt.squeeze(), self.cfg)
        y_pred_mel = extract_mel_features(
            y_pred.squeeze(), self.cfg
        )

        loss = self.l1Loss(y_gt_mel, y_pred_mel) * 45

        return loss