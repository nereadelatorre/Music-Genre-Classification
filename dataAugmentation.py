import random
import torch


class SpecAugment(object):
    """
    Mètode senzill d'augmentació per reconeixement automàtic de veu.
    Aplica màscares de freqüència, màscares de temps i escalat d'amplitud.
    """

    def __init__(self,
                 freq_mask_param=10,      # Amplada màxima de la màscara de freqüència
                 time_mask_param=20,      # Amplada màxima de la màscara de temps
                 num_freq_masks=1,        # Nombre de màscares de freqüència a aplicar
                 num_time_masks=1,        # Nombre de màscares de temps a aplicar
                 amplitude_scale_range=(0.8, 1.2),  # Rang d'escalat aleatori de l'amplitud
                 apply_prob=0.5):         # Probabilitat d'aplicar cada augmentació
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.amplitude_scale_range = amplitude_scale_range
        self.apply_prob = apply_prob

    def __call__(self, mel_spectrogram):
        """
        Aplica augmentacions a un espectrograma de Mel.
        Args:
            mel_spectrogram: Tensor de forma [temps, freqüència]
        Retorna:
            Espectrograma augmentat amb la mateixa forma
        """
        mel = mel_spectrogram.clone()
        time_steps, freq_bins = mel.shape

        # Aplicar màscara de freqüència
        if random.random() < self.apply_prob:
            for _ in range(self.num_freq_masks):
                f = random.randint(0, self.freq_mask_param)
                if f > 0:
                    f0 = random.randint(0, freq_bins - f)
                    mel[:, f0:f0 + f] = 0.0

        # Aplicar màscara de temps
        if random.random() < self.apply_prob:
            for _ in range(self.num_time_masks):
                t = min(random.randint(0, self.time_mask_param), time_steps // 4)
                if t > 0:
                    t0 = random.randint(0, time_steps - t)
                    mel[t0:t0 + t, :] = 0.0

        # Aplicar escalat d'amplitud
        if random.random() < self.apply_prob:
            scale = random.uniform(*self.amplitude_scale_range)
            mel = mel * scale

        return mel


class RandomShift(object):
    """
    Desplaça aleatòriament l'eix temporal de l'espectrograma de Mel de forma circular.
    """

    def __init__(self, max_shift_percent=0.2, apply_prob=0.5):
        self.max_shift_percent = max_shift_percent
        self.apply_prob = apply_prob

    def __call__(self, mel_spectrogram):
        if random.random() < self.apply_prob:
            time_steps = mel_spectrogram.shape[0]
            max_shift = int(time_steps * self.max_shift_percent)
            if max_shift > 0:
                shift = random.randint(-max_shift, max_shift)
                if shift != 0:
                    mel_spectrogram = torch.roll(mel_spectrogram, shifts=shift, dims=0)
        return mel_spectrogram


class RandomStretch(object):
    """
    Estira o comprimeix aleatòriament l'eix temporal de l'espectrograma,
    simulant canvis de tempo.
    """

    def __init__(self, stretch_range=(0.8, 1.2), apply_prob=0.5):
        self.stretch_range = stretch_range
        self.apply_prob = apply_prob

    def __call__(self, mel_spectrogram):
        if random.random() < self.apply_prob:
            time_steps, freq_bins = mel_spectrogram.shape
            stretch_factor = random.uniform(*self.stretch_range)
            new_time_steps = int(time_steps * stretch_factor)
            if new_time_steps <= 0:
                return mel_spectrogram

            indices = torch.linspace(0, time_steps - 1, new_time_steps).clamp(0, time_steps - 1)
            idx_low = indices.long()
            idx_high = (idx_low + 1).clamp(0, time_steps - 1)
            frac = indices - idx_low.float()

            mel_stretched = torch.zeros(new_time_steps, freq_bins)
            for i in range(new_time_steps):
                weight_low = 1.0 - frac[i]
                weight_high = frac[i]
                mel_stretched[i] = weight_low * mel_spectrogram[idx_low[i]] + weight_high * mel_spectrogram[idx_high[i]]

            return mel_stretched

        return mel_spectrogram


class AddNoise(object):
    """
    Afegeix soroll aleatori a l'espectrograma de Mel.
    """

    def __init__(self, noise_level=0.005, apply_prob=0.5):
        self.noise_level = noise_level
        self.apply_prob = apply_prob

    def __call__(self, mel_spectrogram):
        if random.random() < self.apply_prob:
            noise = torch.randn_like(mel_spectrogram) * self.noise_level
            return mel_spectrogram + noise
        return mel_spectrogram


class FrequencyShift(object):
    """
    Desplaça les bandes de freqüència amunt o avall.
    """

    def __init__(self, max_shift=4, apply_prob=0.5):
        self.max_shift = max_shift
        self.apply_prob = apply_prob

    def __call__(self, mel_spectrogram):
        if random.random() < self.apply_prob:
            shift = random.randint(-self.max_shift, self.max_shift)
            if shift != 0:
                mel_spectrogram = torch.roll(mel_spectrogram, shifts=shift, dims=1)

                # Netejar les bandes desplaçades que han estat "envoltades"
                if shift > 0:
                    mel_spectrogram[:, :shift] = 0.0
                else:
                    mel_spectrogram[:, shift:] = 0.0

        return mel_spectrogram


class ComposedAugmentation:
    """
    Composa múltiples transformacions d’augmentació.
    """

    def __init__(self, transforms=None, apply_prob=0.8):
        self.transforms = transforms or []
        self.apply_prob = apply_prob

    def __call__(self, mel_spectrogram):
        if random.random() < self.apply_prob:
            for t in self.transforms:
                mel_spectrogram = t(mel_spectrogram)
        return mel_spectrogram


def get_audio_transforms(augment=True):
    """
    Crea una composició de transformacions per a entrenament.

    Args:
        augment: si False, retorna una transformació buida

    Retorna:
        ComposedAugmentation amb transformacions configurades
    """
    if not augment:
        return None

    transforms_list = [
        SpecAugment(
            freq_mask_param=10,
            time_mask_param=20,
            num_freq_masks=2,
            num_time_masks=2,
            apply_prob=0.5
        ),
        RandomShift(max_shift_percent=0.15, apply_prob=0.3),
        AddNoise(noise_level=0.003, apply_prob=0.3),
        FrequencyShift(max_shift=2, apply_prob=0.3),
    ]

    return ComposedAugmentation(transforms_list, apply_prob=0.8)
