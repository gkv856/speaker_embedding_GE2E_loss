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

        hp = self.hp

        # list of folders (speakers) in the folder
        audio_path = os.path.join(hp.general.project_root, hp.raw_audio.raw_audio_path)

        lst_all_speaker_folders = os.listdir(audio_path)

        if self.verbose:
            print("Text independent speaker verification (TISV) utterance feature extraction started..")
            print(f"Total speakers to be saved {len(lst_all_speaker_folders)}")
        # looping through each speaker
        for i, folder in enumerate(lst_all_speaker_folders):

            # path of each speaker
            per_speaker_folder = os.path.join(audio_path, folder)
            per_speaker_wavs = os.listdir(per_speaker_folder)

            if self.verbose:
                print(f"\nProcessing speaker no. {i + 1} with '{len(per_speaker_wavs)}'' audio files")

            # looping through all the folders for a given speaker
            for utter_wav_file in per_speaker_wavs:
                # path of each utterance
                utter_wav_file_path = os.path.join(per_speaker_folder, utter_wav_file)

                # open the individual audio file and load it as a np array
                # Split the utterance into partials and forward them through the model
                mel_spects = wav_to_mel_spectrogram(utter_wav_file_path, hp)

                # saving mel-spectrogram as train and test splits for Embedding model
                spkr_id = utter_wav_file.split(".")[0]
                self.__save_mel_spects_as_train_test_split(mel_spects, folder, spkr_id)

                # saving mel-spectrogram as training data for AutoVC model
                self.__save_mel_spects_as_train_test_split(mel_spects, folder, spkr_id, split=False)

        print("Spectrograms saved!!")

    def __save_mel_spects_as_train_test_split(self, mel_spects, folder, spkr_id, split=True):
        """
        this function randomly splits the mel_spects and saves a random section of the spect as a
        train and test data for Embedding model
        Also, when split is False then it saves the entire spect as Train data for AutoVC model

        Finally saves them as numpy files
        :param mel_spects: mel spectrogram of shape AxB
        :param spec_path_train: path to save train data
        :param spec_path_test:  path to save test data
        :param folder: folder name
        :param i: dataset count
        :return:
        """
        hp = self.hp
        if split:

            # Create folder if does not exist, if exist then ignore
            train_path = os.path.join(hp.general.project_root, hp.m_ge2e.tt_data.train_spects_path)
            test_path = os.path.join(hp.general.project_root, hp.m_ge2e.tt_data.test_spects_path)
            os.makedirs(train_path, exist_ok=True)
            os.makedirs(test_path, exist_ok=True)

            utter_mel_len = mel_spects.shape[0]
            test_percent = 1 - hp.m_ge2e.tt_data.train_percent
            test_len = int(utter_mel_len * test_percent)

            # ensuring min test size
            if test_len < hp.m_ge2e.tt_data.min_test_utter_len:
                test_len = hp.m_ge2e.tt_data.min_test_utter_len

            # this will give us a random starting point from where we can clip the test audio
            # we dont start from 0 therefore p1 can be designed as p1 = mel_spects[0: idx, :]
            idx = random.randint(1, utter_mel_len - test_len - 1)
            test_mel_spects = mel_spects[idx:idx + test_len, :]

            # collecting train data
            p1 = mel_spects[0: idx, :]
            p2 = mel_spects[idx + test_len:, :]
            train_mel_spects = np.concatenate((p1, p2), axis=0)

            # creating train folders and ignoring if exist
            train_dir_path = os.path.join(train_path, folder)
            os.makedirs(train_dir_path, exist_ok=True)

            # creating test folders and ignoring if exist
            test_dir_path = os.path.join(test_path, folder)
            os.makedirs(test_dir_path, exist_ok=True)

            # now saving the numpy files
            file_full_path = os.path.join(train_dir_path, f"sv_{spkr_id}.npy")
            np.save(file_full_path, train_mel_spects)

            if self.verbose:
                print(f"'m_Embed: Training data' Size saved = {train_mel_spects.shape}")

            file_full_path = os.path.join(test_dir_path, f"sv_{spkr_id}.npy")
            np.save(file_full_path, test_mel_spects)

            if self.verbose:
                print(f"'m_Embed: Eval data' Size saved = {test_mel_spects.shape}")

        else:
            # AutoVC model calculates loss on self construction therefore no training data required
            # Create folder if does not exist, if exist then ignore
            train_path = os.path.join(hp.general.project_root, hp.m_avc.tt_data.train_spects_path)
            os.makedirs(train_path, exist_ok=True)

            # creating train folders and ignoring if exist
            train_dir_path = os.path.join(train_path, folder)
            os.makedirs(train_dir_path, exist_ok=True)

            # now saving the numpy files
            file_full_path = os.path.join(train_dir_path, f"sv_{spkr_id}.npy")
            np.save(file_full_path, mel_spects)
            if self.verbose:
                print(f"'m_AVC: Training data' Size saved = {mel_spects.shape}")


# quick test, below code will not be executed when the file is imported
# it runs only when this file is directly executed
if __name__ == '__main__':
    from strings.constants import hp

    cr_obj = CreateSpectrogram(hp)
    cr_obj.save_spectrogram_tisv()
