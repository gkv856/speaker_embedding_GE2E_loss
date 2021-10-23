import os

from s3_auto_voice_cloner.s7_mel_audio_setup import get_wave_net_model, convert_mel_specs_to_audio
from utils.audio_utils import AudioUtils
from strings.constants import hp

import soundfile as sf

au = AudioUtils(hp)

wav_path = "static/raw_data/wavs/p225/p225_003.wav"
wav_path = os.path.join(hp.general.project_root, wav_path)
mel_spects = au.get_mel_spects_from_audio(wav_path)
print(mel_spects.shape)


# slicing the audio to smaller portion, you can use the whole audio mel as well
c = mel_spects[0]#, :, :]
print(c.shape)

# getting wavenet model, this will up sample the mel-spec to final audio
wave_net_model, wave_net_hp = get_wave_net_model(hp)


waveform = convert_mel_specs_to_audio(wave_net_model, wave_net_hp, hp, mel_specs=c)
sf.write('reconstructed_audio.wav', waveform, 16000, 'PCM_24')
print(1)

# ok
# ok2
# ok3