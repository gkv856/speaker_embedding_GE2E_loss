"""
Step7 of Auto Voice Cloning is to: create audio from cloned voice mel-spectrogram using wavenet pre-trained model
"""
from s3_auto_voice_cloner.s7_mel_audio_setup import get_wave_net_model, convert_mel_specs_to_audio
from strings.constants import hp
import pickle
import soundfile as sf
import os
import numpy as np


# getting wavenet model, this will up sample the mel-spec to final audio
from utils.audio_utils import preprocess_wav, compute_partial_slices, wav_to_mel_spectrogram

wave_net_model, wave_net_hp = get_wave_net_model(hp)


path = "static/raw_data/wavs/p225/p225_003.wav"

wav = preprocess_wav(path, hp=hp)
wav_slices, mel_slices = compute_partial_slices(len(wav), hp=hp)
max_wave_length = wav_slices[-1].stop
if max_wave_length >= len(wav):
    wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

# Split the utterance into partials and forward them through the model
mel = wav_to_mel_spectrogram(wav, hp)

# now the the audio->np array-> mel-spectrogram
mels = np.array([mel[s] for s in mel_slices])
utter_1 = mels[1, :, :]
waveform = convert_mel_specs_to_audio(wave_net_model, wave_net_hp, hp, mel_specs=utter_1)
sf.write('name_test.wav', waveform, 16000, 'PCM_24')
print(1)



#
# # looping over each mel-spec and save it as an audio
# for vc_cross_mel_spec in vc_cross_mel_specs:
#     name = vc_cross_mel_spec[0]
#     mel_specs = vc_cross_mel_spec[1]
#     print(name)
#     waveform = convert_mel_specs_to_audio(wave_net_model, wave_net_hp, mel_specs=mel_specs)
#     sf.write(name + '.wav', waveform, 16000, 'PCM_24')
#
# print(7)


import os
import pickle
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
from numpy.random import RandomState


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def pySTFT(x, fft_length=1024, hop_length=256):
    x = np.pad(x, int(fft_length // 2), mode='reflect')

    noverlap = fft_length - hop_length
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // hop_length, fft_length)
    strides = x.strides[:-1] + (hop_length * x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)

    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T

    return np.abs(result)


mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, 16000, order=5)

# audio file directory
rootDir = './wavs'
# spectrogram directory
targetDir = './spmel'

dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

for subdir in sorted(subdirList):
    print(subdir)
    if not os.path.exists(os.path.join(targetDir, subdir)):
        os.makedirs(os.path.join(targetDir, subdir))
    _, _, fileList = next(os.walk(os.path.join(dirName, subdir)))
    prng = RandomState(int(subdir[1:]))
    for fileName in sorted(fileList):
        # Read audio file
        x, fs = sf.read(os.path.join(dirName, subdir, fileName))
        # Remove drifting noise
        y = signal.filtfilt(b, a, x)
        # Ddd a little random noise for model roubstness
        wav = y * 0.96 + (prng.rand(y.shape[0]) - 0.5) * 1e-06
        # Compute spect
        D = pySTFT(wav).T
        # Convert to mel and normalize
        D_mel = np.dot(D, mel_basis)
        D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
        S = np.clip((D_db + 100) / 100, 0, 1)
        # save spect
        np.save(os.path.join(targetDir, subdir, fileName[:-4]),
                S.astype(np.float32), allow_pickle=False)