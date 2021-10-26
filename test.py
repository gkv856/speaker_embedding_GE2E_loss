import tor


class AudioUtils:
    def __init__(self, hp):
        self.hp = hp

        tfm_to_spect = ta.transforms.Spectrogram(n_fft=800)