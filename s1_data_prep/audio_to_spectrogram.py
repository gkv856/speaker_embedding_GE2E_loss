import os
import random

import numpy as np

from utils.audio_utils import wav_to_mel_spectrogram


class CreateSpectrogram:
    def __init__(self, hp, verbose=False):
        self.hp = hp
        self.verbose = verbose

    def save_spectrogram_tisv(self):
        """
        This method creates mel-spectrogram from the raw audio and then saves for Embedding model and AutoVC model
        as train and test dataset.
        Note: AutoVC model works on self construction therefore no separate test data is required
        :param hp:
        :return:
        """

        # list of folders (speakers) in the folder
        audio_path = os.path.join(self.hp.general.project_root, self.hp.raw_audio.raw_audio_path)

        lst_all_speaker_folders = os.listdir(audio_path)

        if self.verbose:
            print("Text independent speaker verification (TISV) utterance feature extraction started..")
            print(f"Total speakers to be saved {len(lst_all_speaker_folders)}")
        # looping through each speaker
        for i, folder in enumerate(lst_all_speaker_folders):
            # path of each speaker
            per_speaker_folder = os.path.join(audio_path, folder)
            per_speaker_wavs = os.listdir(per_speaker_folder)

            random.shuffle(per_speaker_wavs)

            # train test data split
            train_data = int(len(per_speaker_wavs) * self.hp.m_ge2e.tt_data.train_percent)

            # save training data
            lst_train = per_speaker_wavs[:train_data]
            self.__save_spect_data(lst_train, folder, per_speaker_folder, for_emb=True, training=True)

            # save test data
            lst_test = per_speaker_wavs[train_data:]
            self.__save_spect_data(lst_test, folder, per_speaker_folder, for_emb=True, training=False)

        print("Spectrograms saved!!")

    def __save_spect_data(self, lst_data, folder, per_speaker_folder, for_emb=True, training=True):

        if self.verbose:
            print(f"\nProcessing speaker '{folder}' with '{len(lst_data)}' audio files")

        # looping through all the folders for a given speaker
        for cnt, utter_wav_file in enumerate(lst_data):
            # path of each utterance
            utter_wav_file_path = os.path.join(per_speaker_folder, utter_wav_file)

            # if self.verbose:
            #     print(f"File '{utter_wav_file_path}'")

            # open the individual audio file and load it as a np array
            # Split the utterance into partials and forward them through the model
            mel_spects = wav_to_mel_spectrogram(utter_wav_file_path, self.hp)
            # print(mel_spects.shape)
            # saving mel-spectrogram as train data for Embedding model
            spkr_id = utter_wav_file.split(".")[0] + f"_{cnt + 1}"
            self.__save_mel_spects(mel_spects, folder, spkr_id, for_emb=for_emb, training=training)

            # saving mel-spectrogram as training data for AutoVC model
            self.__save_mel_spects(mel_spects, folder, spkr_id, for_emb=False)

    def __save_mel_spects(self, mel_spects, folder, spkr_id, for_emb=True, training=True):

        if for_emb:
            if training:
                dir_path = os.path.join(self.hp.general.project_root, self.hp.m_ge2e.tt_data.train_spects_path)
            else:
                dir_path = os.path.join(self.hp.general.project_root, self.hp.m_ge2e.tt_data.test_spects_path)

            self.__save_mel_data(folder, mel_spects, spkr_id, dir_path, m_name="GE2E")

        else:
            # AutoVC model calculates loss on self construction therefore no training data required
            # Create folder if does not exist, if exist then ignore
            dir_path = os.path.join(self.hp.general.project_root, self.hp.m_avc.tt_data.train_spects_path)
            self.__save_mel_data(folder, mel_spects, spkr_id, dir_path, m_name="AutoVC")

    def __save_mel_data(self, folder, mel_spects, spkr_id, dir_path, m_name="GE2E"):
        # Create folder if does not exist, if exist then ignore
        os.makedirs(dir_path, exist_ok=True)

        # creating data folders and ignoring if exist
        data_dir_path = os.path.join(dir_path, folder)
        os.makedirs(data_dir_path, exist_ok=True)

        # now saving the numpy files
        file_full_path = os.path.join(data_dir_path, f"sv_{spkr_id}.npy")
        np.save(file_full_path, mel_spects)

        if self.verbose:
            print(f"'{m_name}: Spectrogram saved and size = {mel_spects.shape}")


# quick test, below code will not be executed when the file is imported
# it runs only when this file is directly executed
if __name__ == '__main__':
    pass
    # from strings.constants import hp
    #
    # cr_obj = CreateSpectrogram(hp, verbose=True)
    # cr_obj.save_spectrogram_tisv()
