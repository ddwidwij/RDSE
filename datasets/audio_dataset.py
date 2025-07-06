
from os.path import join
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
from torchaudio import load
import numpy as np
import torch.nn.functional as F


def get_window(window_type, window_length):
    if window_type == 'sqrthann':
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    elif window_type == 'hann':
        return torch.hann_window(window_length, periodic=True)
    else:
        raise NotImplementedError(f"Window type {window_type} not implemented!")


class Specs(Dataset):
    def __init__(self, data_dir, subset, shuffle_spec, num_frames=256,
            format='De-noising', normalize="noisy", spec_transform=None,padding=None,
            stft_kwargs=None):

        # Read file paths according to file naming format.
        if format == "De-noising":
            self.clean_files = sorted(glob(join(data_dir, subset) + '/clean/*.wav'))
            self.noisy_files = sorted(glob(join(data_dir, subset) + '/noisy/*.wav'))
        elif format == "De-reverb":
            self.clean_files = sorted(glob(join(data_dir, subset) + '/anechoic/*.wav'))
            self.noisy_files = sorted(glob(join(data_dir, subset) + '/reverb/*.wav'))
        else:
            # Feel free to add your own directory format
            raise NotImplementedError(f"Directory format {format} unknown!")

        self.num_frames = num_frames
        self.shuffle_spec = shuffle_spec
        self.normalize = normalize
        self.spec_transform = spec_transform
        self.padding = padding

        assert all(k in stft_kwargs.keys() for k in ["n_fft", "hop_length", "center", "window"]), "misconfigured STFT kwargs"
        self.stft_kwargs = stft_kwargs
        self.hop_length = self.stft_kwargs["hop_length"]
        assert self.stft_kwargs.get("center", None) == True, "'center' must be True for current implementation"

    def __getitem__(self, i):
        x, _ = load(self.clean_files[i])
        y, _ = load(self.noisy_files[i])

        if self.padding:
           # formula applies for center=True
           target_len = (self.num_frames - 1) * self.hop_length
           current_len = x.size(-1)
           pad = max(target_len - current_len, 0)
           if pad == 0:
               # extract random part of the audio file
               if self.shuffle_spec:
                   start = int(np.random.uniform(0, current_len - target_len))
               else:
                   start = int((current_len - target_len) / 2)
               x = x[..., start:start + target_len]
               y = y[..., start:start + target_len]
           else:
               # pad audio if the length T is smaller than num_frames
               x = F.pad(x, (pad // 2, pad // 2 + (pad % 2)), mode='constant')
               y = F.pad(y, (pad // 2, pad // 2 + (pad % 2)), mode='constant')

        # normalize w.r.t to the noisy or the clean signal or not at all
        # to ensure same clean signal power in x and y.
        if self.normalize == "noisy":
            normfac = y.abs().max()
        elif self.normalize == "clean":
            normfac = x.abs().max()
        elif self.normalize == "not":
            normfac = 1.0
        x = x / normfac
        y = y / normfac

        X = torch.stft(x, **self.stft_kwargs)
        Y = torch.stft(y, **self.stft_kwargs)

        X, Y = self.spec_transform(X), self.spec_transform(Y)
        return X, Y

    def __len__(self):
         return len(self.clean_files)

    def load_other(self, i):
        file_name = self.noisy_files[i]
        y, _ = load(self.noisy_files[i])
        norm_factor = y.abs().max()
        T_orig = y.size(1)
        return file_name.split("/")[-1], norm_factor, T_orig
class SpecsDataModule(pl.LightningDataModule):
    def __init__(
            self, base_dir, format='De-noising', batch_size=8,
            n_fft=510, hop_length=128, num_frames=256, window='hann',
            num_workers=4, dummy=False, spec_factor=0.15, spec_abs_exponent=0.5,
            gpu=True, normalize='noisy', transform_type="exponent", **kwargs
    ):
        super().__init__()
        self.base_dir = base_dir
        self.format = format
        self.batch_size = batch_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.window = get_window(window, self.n_fft)
        self.windows = {}
        self.num_workers = num_workers
        self.dummy = dummy
        self.spec_factor = spec_factor
        self.spec_abs_exponent = spec_abs_exponent
        self.gpu = gpu
        self.normalize = normalize
        self.transform_type = transform_type
        self.kwargs = kwargs

    def setup(self, stage=None):
        specs_kwargs = dict(
            stft_kwargs=self.stft_kwargs, num_frames=self.num_frames,
            spec_transform=self.spec_fwd, **self.kwargs
        )
        if stage == 'fit' or stage is None:
            self.train_set = Specs(data_dir=self.base_dir, subset='train',
                                   shuffle_spec=True, format=self.format,
                                   normalize=self.normalize, padding=True, **specs_kwargs)
            self.valid_set = Specs(data_dir=self.base_dir, subset='valid',
                                   shuffle_spec=True, format=self.format,
                                   normalize=self.normalize, padding=True, **specs_kwargs)
        if stage == 'test' or stage is None:
            self.test_set = Specs(data_dir=self.base_dir, subset='test',
                                   shuffle_spec=False, format=self.format,padding=False,
                                  normalize=self.normalize, **specs_kwargs)

    def spec_fwd(self, spec):
        if self.transform_type == "exponent":
            if self.spec_abs_exponent != 1:
                # only do this calculation if spec_exponent != 1, otherwise it's quite a bit of wasted computation
                # and introduced numerical error
                e = self.spec_abs_exponent
                spec = spec.abs() ** e * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "log":
            spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "none":
            spec = spec
        return spec

    def spec_back(self, spec):
        if self.transform_type == "exponent":
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs() ** (1 / e) * torch.exp(1j * spec.angle())
        elif self.transform_type == "log":
            spec = spec / self.spec_factor
            spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
        elif self.transform_type == "none":
            spec = spec
        return spec

    @property
    def stft_kwargs(self):
        return {**self.istft_kwargs, "return_complex": True}

    @property
    def istft_kwargs(self):
        return dict(
            n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, center=True
        )

    def _get_window(self, x):
        """
        Retrieve an appropriate window for the given tensor x, matching the device.
        Caches the retrieved windows so that only one window tensor will be allocated per device.
        """
        window = self.windows.get(x.device, None)
        if window is None:
            window = self.window.to(x.device)
            self.windows[x.device] = window
        return window

    def stft(self, sig):
        window = self._get_window(sig)
        return torch.stft(sig, **{**self.stft_kwargs, "window": window})

    def istft(self, spec, length=None):
        window = self._get_window(spec)
        return torch.istft(spec, **{**self.istft_kwargs, "window": window, "length": length})

