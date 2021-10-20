from s1_data_prep.audio_to_spectrogram import save_spectrogram_tisv
from strings.constants import hp


# step 1 prepare the spectrogram from the raw audio file
save_spectrogram_tisv(hp, speaker_utter_cnt=65)

