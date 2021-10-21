"""
Step6 of Auto Voice Cloning is to: create cloned voices mel-spectrogram
using trained AutoVC model, we pass audio from one speaker and embedding of a different speaker to create new
mel-spectrogram
"""
from s3_auto_voice_cloner.s6_create_cross_speaker_mel_specs import create_mel_specs_per_speaker
from strings.constants import hp

create_mel_specs_per_speaker(hp)

print(6)
