"""
Step1 of the Auto voice cloning is to: create mel-spectrogram from the audio files.
This will be used to get a user's embedding or 256 dim vector representing user's speaking style
"""
from s1_data_prep.audio_to_spectrogram import CreateSpectrogram
from strings.constants import hp


# step 1 prepare the spectrogram from the raw audio file
cr_obj = CreateSpectrogram(hp)
cr_obj.save_spectrogram_tisv()

print(1)

