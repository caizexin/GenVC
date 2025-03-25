from dataclasses import dataclass, field
from typing import List
from coqpit import Coqpit

@dataclass
class BaseVocoderConfig(Coqpit):
    input_feat_dim: int = 1024
    sample_rate: int = 24000
    fft_size: int = 1024
    num_mels: int = 100
    mel_fmin: int = 0
    mel_fmax: int = 12000
    win_length: int = 1024
    hop_length: int = 256
    upsample_initial_channel: int = 256
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    resblock_dilation_sizes: List[List[int]] = field(default_factory=lambda: [[1, 2], [2, 6], [3, 12]])
    upsample_rates: List[int] = field(default_factory=lambda: [8, 8, 4])
    upsample_kernal_sizes: List[int] = field(default_factory=lambda: [16, 16, 8])
    resblock_type: str = "2"
    # MPD discriminator
    mpd_reshapes: List[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])
    mpd_discriminator_channel_mult_factor: int = 1
    mpd_use_spectral_norm: bool = False

    # MSTFT Discriminator
    msstftd_filters: int = 32

    # MSCQT Discriminator
    mssbcqtd_filters: int = 32
    mssbcqtd_max_filters: int = 1024
    mssbcqtd_filters_scale: int = 1
    mssbcqtd_dilations: List[int] = field(default_factory=lambda: [1, 2, 4])
    mssbcqtd_in_channels: int = 1
    mssbcqtd_out_channels: int = 1
    mssbcqtd_hop_lengths: List[int] = field(default_factory=lambda: [512, 256, 256])
    mssbcqtd_n_octavess: List[int] = field(default_factory=lambda: [9, 9, 9])
    mssbcqtd_bins_per_octave: List[int] = field(default_factory=lambda: [24, 36, 48])     


