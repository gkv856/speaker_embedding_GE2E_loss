import random
import struct
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
import webrtcvad
from scipy.ndimage.morphology import binary_dilation

import soundfile as sf
from scipy import signal
from librosa.filters import mel
from numpy.random import RandomState
import os
import pickle

from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
from numpy.random import RandomState

int16_max = (2 ** 15) - 1


def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray], hp, source_sr: Optional[int] = None):
    """
    Applies preprocessing operations to a waveform either on disk or in memory such that
    The waveform will be resampled to match the data hyperparameters.

    :param fpath_or_wav: either a filepath to an audio file (many extensions are supported, not
    just .wav), either the waveform as a numpy array of floats.
    :param source_sr: if passing an audio waveform, the sampling rate of the waveform before
    preprocessing. After preprocessing, the waveform'speaker sampling rate will match the data
    hyperparameters. If passing a filepath, the sampling rate will be automatically detected and
    this argument will be ignored.
    """
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(str(fpath_or_wav), sr=None)
    else:
        wav = fpath_or_wav

    # Resample the wav
    if source_sr is not None:
        wav = librosa.resample(wav, source_sr, hp.audio.sampling_rate)

    # Apply the preprocessing: normalize volume and shorten long silences
    wav = normalize_volume(wav, hp.vad.audio_norm_target_dBFS, increase_only=True)
    wav = trim_long_silences(wav, hp)

    return wav


def trim_long_silences(wav, hp):
    """
    Ensures that segments without voice in the waveform remain no longer than a
    threshold determined by the VAD parameters in params.py.

    :param wav: the raw waveform as a numpy array of floats
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    # Compute the voice detection window size
    samples_per_window = (hp.vad.vad_window_length * hp.audio.sampling_rate) // 1000

    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]

    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))

    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flag = vad.is_speech(pcm_wave[window_start * 2:window_end * 2], sample_rate=hp.audio.sampling_rate)
        voice_flags.append(voice_flag)

    voice_flags = np.array(voice_flags)

    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width

    audio_mask = moving_average(voice_flags, hp.vad.vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)

    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(hp.vad.vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)

    return wav[audio_mask == True]


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = np.sqrt(np.mean((wav * int16_max) ** 2))
    wave_dBFS = 20 * np.log10(rms / int16_max)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))


def compute_partial_slices(n_samples: int, hp):
    """
    Computes where to split an utterance waveform and its corresponding mel spectrogram to
    obtain partial utterances of <partials_n_frames> each. Both the waveform and the
    mel spectrogram slices are returned, so as to make each partial utterance waveform
    correspond to its spectrogram.

    The returned ranges may be indexing further than the length of the waveform. It is
    recommended that you pad the waveform with zeros up to wav_slices[-1].stop.

    :param n_samples: the number of samples in the waveform
    :param rate: how many partial utterances should occur per second. Partial utterances must
    cover the span of the entire utterance, thus the rate should not be lower than the inverse
    of the duration of a partial utterance. By default, partial utterances are 1.6s long and
    the minimum rate is thus 0.625.
    :param min_coverage: when reaching the last partial utterance, it may or may not have
    enough frames. If at least <min_pad_coverage> of <partials_n_frames> are present,
    then the last partial utterance will be considered by zero-padding the audio. Otherwise,
    it will be discarded. If there aren't enough frames for one partial utterance,
    this parameter is ignored so that the function always returns at least one slice.
    :return: the waveform slices and mel spectrogram slices as lists of array slices. Index
    respectively the waveform and the mel spectrogram with these slices to obtain the partial
    utterances.
    """
    assert 0 < hp.vad.min_coverage <= 1

    # Compute how many frames separate two partial utterances
    samples_per_frame = int((hp.audio.sampling_rate * hp.mel_fb.mel_window_step / 1000))
    n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
    frame_step = int(np.round((hp.audio.sampling_rate / hp.vad.rate_partial_slices) / samples_per_frame))

    min_frame_step = (hp.audio.sampling_rate / (samples_per_frame * hp.audio.partials_n_frames))
    assert 0 < frame_step, "The rate is too high"
    assert frame_step <= hp.audio.partials_n_frames, "The rate is too low, it should be %f at least" % min_frame_step

    # Compute the slices
    wav_slices, mel_slices = [], []
    steps = max(1, n_frames - hp.audio.partials_n_frames + frame_step + 1)
    for i in range(0, steps, frame_step):
        mel_range = np.array([i, i + hp.audio.partials_n_frames])
        wav_range = mel_range * samples_per_frame
        mel_slices.append(slice(*mel_range))
        wav_slices.append(slice(*wav_range))

    # Evaluate whether extra padding is warranted or not
    last_wav_range = wav_slices[-1]
    coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
    if coverage < hp.vad.min_coverage and len(mel_slices) > 1:
        mel_slices = mel_slices[:-1]
        wav_slices = wav_slices[:-1]

    return wav_slices, mel_slices


def pySTFT(x, fft_length=1024, hop_length=256):
    """
    this function returns spectrogram
    :param x: np array for the audio file
    :param fft_length: fft length for fast fourier transform (https://www.youtube.com/watch?v=E8HeD-MUrjY)
    :param hop_length: hop_lenght is the sliding overlapping window size normally fft//4 works the best
    :return: spectrogram in the form of np array
    """
    x = np.pad(x, int(fft_length // 2), mode='reflect')

    noverlap = fft_length - hop_length
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // hop_length, fft_length)
    strides = x.strides[:-1] + (hop_length * x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T

    return np.abs(result)


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def wav_to_mel_spectrogram(wav, hp):
    """
    Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    Note: this not a log-mel spectrogram.
    """

    # creating mel basis matrix
    mel_basis = mel(hp.audio.sampling_rate,
                    hp.audio.n_fft,
                    fmin=90,
                    fmax=7600,
                    n_mels=hp.audio.mel_n_channels).T

    min_level = np.exp(-100 / 20 * np.log(10))

    # getting audio as a np array
    pp_wav = preprocess_wav(wav, hp)

    # Compute spect
    spectrogram = pySTFT(pp_wav).T
    # Convert to mel and normalize
    mel_spect = np.dot(spectrogram, mel_basis)
    D_db = 20 * np.log10(np.maximum(min_level, mel_spect)) - 16
    norm_mel_spect = np.clip((D_db + 100) / 100, 0, 1)

    return norm_mel_spect


def shuffle_along_axis(a, axis):
    """
    :param a: nd array e.g. [40, 180, 80]
    :param axis: array axis. e.g. 0
    :return: a shuffled np array along the given axis
    """
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)
