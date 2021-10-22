import random

import numpy as np
import torch.utils.data as data
import os

from utils.audio_utils import shuffle_along_axis


class EmbeddingModelTTDataset(data.Dataset):
    """
        this class will be used to fetch the spectrogram prepared in the step1
        these spectrograms will be used to train speaker verification model using GE2E loss

    """

    def __init__(self, data_path, hp, training=True):

        # getting all the folders/speakers
        data_iter = iter(os.walk(data_path))

        # note: walk returns a tuple
        # 0th item -> name of current folder
        # 1st item -> list of folders in the current folder
        # 2rd item -> files in current folder

        iter_data = next(data_iter)
        data_path = iter_data[0]
        self.lst_speakers = iter_data[1]

        # getting total speakers
        self.data_len = len(self.lst_speakers)

        self.dict_spkr_utters = {}
        for spr in self.lst_speakers:
            d_path = os.path.join(data_path, spr)
            utters = []
            for utter in os.walk(d_path):
                for f in utter[2]:
                    file_path = os.path.join(utter[0], f)
                    utters.append(file_path)

            # storing the absolute path of spectrogram utterances for each speaker
            self.dict_spkr_utters[spr] = utters

        # prints speaker id and np spect count
        # for k, v in self.utter_per_speaker.items():
        #     print(k, len(v))

        # setting number of utterance to fetch per speaker
        if training:
            self.utter_num = hp.m_ge2e.training_M
            self.min_utter_len = hp.m_ge2e.tt_data.min_train_utter_len
            # shuffle the speaker's list
            # if this dataset belongs to training then shuffle the list of speakers
            random.shuffle(self.lst_speakers)

        else:
            self.utter_num = hp.m_ge2e.test_M
            self.min_utter_len = hp.m_ge2e.tt_data.min_test_utter_len

        self.training = training

    def __len__(self):
        return self.data_len

    def __get_utterances(self, utterances, idx):

        selected_spkr = self.lst_speakers[idx]
        lst_utters = self.dict_spkr_utters[selected_spkr]

        # randomly select M utterances per speaker
        utter_idx = np.random.randint(0, len(lst_utters), self.utter_num)

        # we will run a loop as len(utter_idx) to getch M utterances since test data is = self.min_utter_len
        # for train data of arbitary length, we will fetch n utterences of length self.min_utter_len such that
        # x * self.min_utter_len = M

        for idx in utter_idx:

            # if the utterances = self.utter_num then stop the loop
            if len(utterances) >= self.utter_num:
                break

            utter_path = lst_utters[idx]
            utter = np.load(utter_path)

            # checking if utter length is less than min len then we will do some padding
            # but also checking if length is not too small. E.g. is length is 20 then just discard this utterance
            st = 0
            ed = st + self.min_utter_len
            tmp_utter = None
            while utter.shape[0] > 100:
                # if the utterances = self.utter_num then stop the loop
                if len(utterances) >= self.utter_num:
                    break

                if utter.shape[0] < self.min_utter_len:
                    len_pad = self.min_utter_len - utter.shape[0]
                    tmp_utter = np.pad(utter, ((0, len_pad), (0, 0)), 'constant')

                elif utter.shape[0] > self.min_utter_len:
                    tmp_utter = utter[st:ed, :]
                    # update the utter
                elif utter.shape[0] == self.min_utter_len:
                    tmp_utter = utter

                if len(tmp_utter) and tmp_utter.shape[0] == self.min_utter_len:
                    utterances.append(tmp_utter)

                # reducing the current content of utter
                utter = utter[ed:, :]

                st = st + ed
                ed = self.min_utter_len + ed

        # shuffling the utterances
        random.shuffle(utterances)

        return utterances

    def __getitem__(self, idx):
        """
            this method is called everytime we fetch data using this dataset
            this method is called 'batch_size' number of times to produce that many data item
        :param idx: is the index of the speaker.
        :return: spectrogram utterance
        """

        utterances = []
        utterances = self.__get_utterances(utterances, idx)

        utters = np.array(utterances)
        while utters.shape[0] < self.utter_num:
            utterances = self.__get_utterances(utterances, idx)
            utters = np.array(utterances)

        # checking the length of utterances returned for training
        assert utters.shape[0] == self.utter_num
        assert utters.shape[1] == self.min_utter_len

        shuffle_along_axis(utters, axis=0)
        return utters


def get_train_test_data_loader(hp):
    """
        creates data loader for training and testing
    :param hp:
    :param lst_train_data: list of numpy files (np array of spectrograms) created in Step1
    :param lst_test_data: list of numpy files (np array of spectrograms) created in Step1
    :return:
    """

    train_specs_path = os.path.join(hp.general.project_root, hp.m_ge2e.tt_data.train_spects_path)
    test_specs_path = os.path.join(hp.general.project_root, hp.m_ge2e.tt_data.test_spects_path)

    train_dataset = EmbeddingModelTTDataset(hp=hp, training=True, data_path=train_specs_path)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=hp.m_ge2e.training_N,  # number of speakers
                                   shuffle=True,
                                   drop_last=True)

    # TODO implement num_workers
    # num_workers=hp.m_ge2e.training_num_workers,

    test_dataset = EmbeddingModelTTDataset(hp=hp, training=False, data_path=test_specs_path)
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=hp.m_ge2e.test_N,  # number of speakers
                                  shuffle=False,
                                  drop_last=True)

    return train_loader, test_loader


# quick test, below code will not be executed when the file is imported
# it runs only when this file is directly executed
if __name__ == '__main__':
    from strings.constants import hp
    import os

    trian_dl, test_dl = get_train_test_data_loader(hp)

    # this will produce 16 items because there are 17 speakers in the training and last is dropped
    for i, mel_db_batch in enumerate(trian_dl):
        print(i, mel_db_batch.shape)

    print("\n##### test data loader####")
    # this will produce 4 items because there are 6 speakers in the test and last is dropped
    for i, mel_db_batch in enumerate(test_dl):
        print(i, mel_db_batch.shape)

    print(1)
