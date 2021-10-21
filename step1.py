"""
Step1 of the Auto voice cloning is to: create mel-spectrogram from the audio files.
This will be used to get a user's embedding or 256 dim vector representing user's speaking style
"""

from s1_data_prep.audio_to_spectrogram import save_spectrogram_tisv
from strings.constants import hp


# step 1 prepare the spectrogram from the raw audio file
save_spectrogram_tisv(hp, speaker_utter_cnt=65)

print(1)

