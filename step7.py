"""
Step7 of Auto Voice Cloning is to: create audio from cloned voice mel-spectrogram using wavenet pre-trained model
"""
import librosa

from s3_auto_voice_cloner.s7_mel_audio_setup import get_wave_net_model, convert_mel_specs_to_audio
from strings.constants import hp
import pickle
import soundfile as sf
import os

# getting wavenet model, this will up sample the mel-spec to final audio
wave_net_model, wave_net_hp = get_wave_net_model(hp)

# reading the cross speaker mel-spectrograms
p1 = os.path.join(hp.general.project_root, hp.m_avc.gen.cross_mel_specs_path)
file_path = os.path.join(p1, hp.m_avc.gen.cross_mel_specs_file)
vc_cross_mel_specs = pickle.load(open(file_path, 'rb'))

from tqdm import tqdm
# looping over each mel-spec and save it as an audio
for vc_cross_mel_spec in vc_cross_mel_specs:
    name = vc_cross_mel_spec[0]
    mel_specs = vc_cross_mel_spec[1]

    waveform = convert_mel_specs_to_audio(wave_net_model, wave_net_hp, hp, mel_specs=mel_specs)
    sf.write(name + '.wav', waveform, 16000, 'PCM_24')

print(7)
