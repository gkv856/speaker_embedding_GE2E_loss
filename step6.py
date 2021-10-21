"""
Step6 of Auto Voice Cloning is to create cloned voices mel-spectrogram
"""
from s3_auto_voice_cloner.s6_create_cross_speaker_mel_specs import create_mel_specs_per_speaker
from strings.constants import hp

create_mel_specs_per_speaker(hp)

print(6)