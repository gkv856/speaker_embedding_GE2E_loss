import os
import random

import numpy as np

from utils.audio_utils import AudioUtils


class CreateSpectrogram:
    def __init__(self, hp, verbose=False):
        self.hp = hp
        self.verbose = verbose
        self.au = AudioUtils(hp)

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

            if self.verbose:
                print(f"\nProcessing speaker '{folder}' with '{len(per_speaker_wavs)}' audio files")

            # placeholder utterances np array
            utterances = np.ndarray((1, 1, 1))

            # looping through all the folders for a given speaker
            for cnt, utter_wav_file in enumerate(per_speaker_wavs):
                # path of each utterance
                utter_wav_file_path = os.path.join(per_speaker_folder, utter_wav_file)

                # if self.verbose:
                #     print(f"File '{utter_wav_file_path}'")

                # open the individual audio file and load it as a np array
                # Split the utterance into partials and forward them through the model
                mel_spects = self.au.get_mel_spects_from_audio(utter_wav_file_path)
                if cnt == 0:
                    utterances = mel_spects
                else:
                    utterances = np.concatenate((utterances, mel_spects), axis=0)

            # shuffling the utterances
            utterances = self.au.shuffle_along_axis(utterances, axis=0)

            # train test data split
            train_data = int(utterances.shape[0] * self.hp.m_ge2e.tt_data.train_percent)

            # save training data
            utter_train = utterances[:train_data, :, :]
            utter_test = utterances[train_data:, :, :]

            # saving training data
            self.__save_mel_spects(utter_train, folder, for_emb=True, training=True)

            # saving test data
            self.__save_mel_spects(utter_test, folder, for_emb=True, training=False)

            # saving mel-spectrogram as training data for AutoVC model
            self.__save_mel_spects(utterances, folder, for_emb=False)

        print("Spectrograms saved!!")

    def __save_mel_spects(self, mel_spects, folder, for_emb=True, training=True):
        purpose = "train"
        m_name = "GE2E"
        if for_emb:
            if training:
                dir_path = os.path.join(self.hp.general.project_root, self.hp.m_ge2e.tt_data.train_spects_path)
            else:
                purpose = "test"
                dir_path = os.path.join(self.hp.general.project_root, self.hp.m_ge2e.tt_data.test_spects_path)

        else:
            # AutoVC model calculates loss on self construction therefore no training data required
            # Create folder if does not exist, if exist then ignore
            dir_path = os.path.join(self.hp.general.project_root, self.hp.m_avc.tt_data.train_spects_path)
            m_name = "AutoVC"

        self.__save_mel_data(folder, mel_spects, dir_path, m_name=m_name, purpose=purpose)

    def __save_mel_data(self, folder, mel_spects, dir_path, m_name="GE2E", purpose="train"):
        # Create folder if does not exist, if exist then ignore
        os.makedirs(dir_path, exist_ok=True)

        # # creating data folders and ignoring if exist
        # data_dir_path = os.path.join(dir_path, folder)
        # os.makedirs(data_dir_path, exist_ok=True)

        # now saving the numpy files
        file_full_path = os.path.join(dir_path, f"sv_{folder}.npy")
        np.save(file_full_path, mel_spects)

        if self.verbose:
            print(f"'{m_name}_{purpose}: Spectrogram saved and size = {mel_spects.shape}")


# quick test, below code will not be executed when the file is imported
# it runs only when this file is directly executed
if __name__ == '__main__':
    pass
    from strings.constants import hp

    hp.raw_audio.raw_audio_path = "static/raw_data/wavs"
    cr_obj = CreateSpectrogram(hp, verbose=False)
    cr_obj.save_spectrogram_tisv()
