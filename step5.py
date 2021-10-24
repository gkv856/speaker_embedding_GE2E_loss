"""
Step6 of Auto Voice Cloning is to: create cloned voices mel-spectrogram
using trained AutoVC model, we pass audio from one speaker and embedding of a different speaker to create new
mel-spectrogram
"""
import tqdm

from s3_auto_voice_cloner.s6_create_cross_speaker_mel_specs import VoiceCloner
from strings.constants import hp
import soundfile as sf
import os

hp.m_avc.gen.best_model_path = "static/model_chk_pts/autovc/AVC_final_1000.pth"
vcs_obj = VoiceCloner(hp, tqdm)

path_audio = "static/raw_data/wavs/p225/p225_003.wav"
path_audio = os.path.join(hp.general.project_root, path_audio)
spkr_p225_mel_spec = vcs_obj.au.get_mel_spects_from_audio(path_audio, partial_slices=False)

path_audio = "static/raw_data/wavs/p226/p226_003.wav"
path_audio = os.path.join(hp.general.project_root, path_audio)
spkr_p226_mel_spec = vcs_obj.au.get_mel_spects_from_audio(path_audio, partial_slices=False)

avc_mel_specs = vcs_obj.create_cross_spkr_mel_spects("p225", "p226", spkr_p225_mel_spec[:320, :])

np_audio = vcs_obj.convert_mel_specs_to_audio(avc_mel_specs)

sf.write('test.wav', np_audio, hp.audio.sampling_rate, 'PCM_24')

print(1)