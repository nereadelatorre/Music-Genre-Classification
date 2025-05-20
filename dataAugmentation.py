import matplotlib.pyplot as plt
from models.baseline import *
from models.CNNLSTM import *
import random


from feature_extraction import load_audio, normalize_audio, trim_silence

class SpecAugment(object):
    """
    Implementation of SpecAugment for mel spectrograms:
    - Time warping (not implemented as it's computationally expensive)
    - Frequency masking
    - Time masking
    - Amplitude scaling

    Reference:
    SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
    https://arxiv.org/abs/1904.08779
    """

    def __init__(self,
                 freq_mask_param=10,
                 time_mask_param=20,
                 num_freq_masks=1,
                 num_time_masks=1,
                 amplitude_scale_range=(0.8, 1.2),
                 apply_prob=0.5):
        """
        Args:
            freq_mask_param: maximum width of frequency mask
            time_mask_param: maximum width of time mask
            num_freq_masks: number of frequency masks to apply
            num_time_masks: number of time masks to apply
            amplitude_scale_range: range for random amplitude scaling
            apply_prob: probability of applying each augmentation
        """
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.amplitude_scale_range = amplitude_scale_range
        self.apply_prob = apply_prob

    def __call__(self, mel_spectrogram):
        """
        Args:
            mel_spectrogram: Tensor of shape [time, freq]
        Returns:
            Augmented mel spectrogram of same shape
        """
        # Always work on a copy to avoid modifying the original
        mel = mel_spectrogram.clone()

        # Get dimensions
        time_steps = mel.shape[0]
        freq_bins = mel.shape[1]

        # Apply frequency masking
        if random.random() < self.apply_prob:
            for i in range(self.num_freq_masks):
                f = random.randint(0, self.freq_mask_param)
                if f > 0:
                    f0 = random.randint(0, freq_bins - f)
                    mel[:, f0:f0 + f] = 0.0

        # Apply time masking
        if random.random() < self.apply_prob:
            for i in range(self.num_time_masks):
                t = min(random.randint(0, self.time_mask_param), time_steps // 4)  # Limit to 1/4 of length
                if t > 0:
                    t0 = random.randint(0, time_steps - t)
                    mel[t0:t0 + t, :] = 0.0

        # Apply amplitude scaling
        if random.random() < self.apply_prob:
            scale = random.uniform(*self.amplitude_scale_range)
            mel = mel * scale

        return mel


class RandomShift(object):
    """
    Randomly shifts the time axis of the mel spectrogram,
    wrapping around to create a circular shift.
    """

    def __init__(self, max_shift_percent=0.2, apply_prob=0.5):
        """
        Args:
            max_shift_percent: maximum shift as a percentage of total time
            apply_prob: probability of applying the shift
        """
        self.max_shift_percent = max_shift_percent
        self.apply_prob = apply_prob

    def __call__(self, mel_spectrogram):
        if random.random() < self.apply_prob:
            time_steps = mel_spectrogram.shape[0]
            max_shift = int(time_steps * self.max_shift_percent)
            if max_shift > 0:
                shift = random.randint(-max_shift, max_shift)
                if shift != 0:
                    # Perform circular shift along time axis
                    mel_spectrogram = torch.roll(mel_spectrogram, shifts=shift, dims=0)

        return mel_spectrogram


class RandomStretch(object):
    """
    Randomly stretches or compresses the time axis of the mel spectrogram,
    simulating tempo changes.
    """

    def __init__(self, stretch_range=(0.8, 1.2), apply_prob=0.5):
        """
        Args:
            stretch_range: range of stretch factors
            apply_prob: probability of applying the stretch
        """
        self.stretch_range = stretch_range
        self.apply_prob = apply_prob

    def __call__(self, mel_spectrogram):
        if random.random() < self.apply_prob:
            # Get original dimensions
            time_steps, freq_bins = mel_spectrogram.shape

            # Generate random stretch factor
            stretch_factor = random.uniform(*self.stretch_range)

            # Calculate new time dimension
            new_time_steps = int(time_steps * stretch_factor)
            if new_time_steps <= 0:
                return mel_spectrogram

            # Use interpolation to stretch/compress
            indices = torch.linspace(0, time_steps - 1, new_time_steps)
            indices = indices.clamp(0, time_steps - 1)

            # Get integer and fractional parts
            idx_low = indices.long()
            idx_high = (idx_low + 1).clamp(0, time_steps - 1)
            frac = indices - idx_low.float()

            # Linear interpolation
            mel_stretched = torch.zeros(new_time_steps, freq_bins)
            for i in range(new_time_steps):
                weight_low = 1.0 - frac[i]
                weight_high = frac[i]
                mel_stretched[i] = weight_low * mel_spectrogram[idx_low[i]] + weight_high * mel_spectrogram[idx_high[i]]

            return mel_stretched

        return mel_spectrogram


class AddNoise(object):
    """
    Adds random noise to the mel spectrogram.
    """

    def __init__(self, noise_level=0.005, apply_prob=0.5):
        """
        Args:
            noise_level: standard deviation of the noise
            apply_prob: probability of applying noise
        """
        self.noise_level = noise_level
        self.apply_prob = apply_prob

    def __call__(self, mel_spectrogram):
        if random.random() < self.apply_prob:
            noise = torch.randn_like(mel_spectrogram) * self.noise_level
            return mel_spectrogram + noise
        return mel_spectrogram


class FrequencyShift(object):
    """
    Shifts the frequency bins up or down.
    """

    def __init__(self, max_shift=4, apply_prob=0.5):
        """
        Args:
            max_shift: maximum number of frequency bins to shift
            apply_prob: probability of applying the shift
        """
        self.max_shift = max_shift
        self.apply_prob = apply_prob

    def __call__(self, mel_spectrogram):
        if random.random() < self.apply_prob:
            shift = random.randint(-self.max_shift, self.max_shift)
            if shift != 0:
                # Perform shift along frequency axis
                mel_spectrogram = torch.roll(mel_spectrogram, shifts=shift, dims=1)

                # Zero out the wrapped frequencies
                if shift > 0:
                    mel_spectrogram[:, :shift] = 0.0
                else:
                    mel_spectrogram[:, shift:] = 0.0

        return mel_spectrogram


class ComposedAugmentation:
    """ Composes multiple augmentations together. """

    def __init__(self, transforms=None, apply_prob=0.8):
        """
        Args:
            transforms: list of augmentation transforms
            apply_prob: probability of applying any augmentation
        """
        self.transforms = transforms or []
        self.apply_prob = apply_prob

    def __call__(self, mel_spectrogram):
        if random.random() < self.apply_prob:
            for t in self.transforms:
                mel_spectrogram = t(mel_spectrogram)
        return mel_spectrogram


def get_audio_transforms(augment=True):
    """
    Creates augmentation transforms for training.

    Args:
        augment: whether to use augmentation or return identity transform

    Returns:
        Composed transforms
    """
    if not augment:
        return None

    transforms_list = [
        SpecAugment(
            freq_mask_param=10,  # maximum frequency mask width
            time_mask_param=20,  # maximum time mask width
            num_freq_masks=2,  # apply 2 frequency masks
            num_time_masks=2,  # apply 2 time masks
            apply_prob=0.5
        ),
        RandomShift(max_shift_percent=0.15, apply_prob=0.3),
        AddNoise(noise_level=0.003, apply_prob=0.3),
        FrequencyShift(max_shift=2, apply_prob=0.3),
        # RandomStretch is more complex and might affect the shape, so it's commented out
        # You can enable it if your model handles variable-length inputs well
        # RandomStretch(stretch_range=(0.9, 1.1), apply_prob=0.3),
    ]

    return ComposedAugmentation(transforms_list, apply_prob=0.8)

"""
def test_augmentations(config):
    transforms = get_audio_transforms(augment=True)
    dataset = FilteredMelSpectrogramDataset(config.h5_path)

    mel, label = dataset[0]
    print("Original shape:", mel.shape)

    # Aplicar transformaciones
    augmented = transforms(mel)
    print("Augmented shape:", augmented.shape)

    # Visualizar antes y después
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(mel.numpy(), aspect='auto', origin='lower')
    ax1.set_title('Original')
    ax2.imshow(augmented.numpy(), aspect='auto', origin='lower')
    ax2.set_title('Augmented')
    plt.tight_layout()
    plt.savefig('augmentation_test.png')
    print("Test image saved as 'augmentation_test.png'")
"""