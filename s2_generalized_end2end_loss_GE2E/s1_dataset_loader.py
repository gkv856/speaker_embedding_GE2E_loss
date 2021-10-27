import random

import numpy as np
import torch.utils.data as data
import os


class EmbeddingModelTTDataset(data.Dataset):
    """
        this class will be used to fetch the spectrogram prepared in the step1
        these spectrograms will be used to train speaker verification model using GE2E loss

    """

    def __init__(self, data_path, hp, training=True):

        # data path saved
        self.data_path = data_path

        # getting all the folders/speakers
        data_iter = iter(os.walk(data_path))

        # note: walk returns a tuple
        # 0th item -> name of current folder
        # 1st item -> list of folders in the current folder
        # 2rd item -> files in current folder

        iter_data = next(data_iter)
        self.lst_spkr_np_files = iter_data[2]

        # getting total speakers
        self.data_len = len(self.lst_spkr_np_files)

        # setting number of utterance to fetch per speaker
        if training:
            self.utter_num = hp.m_ge2e.training_M
            self.min_utter_len = hp.m_ge2e.tt_data.min_train_utter_len
            # shuffle the speaker's list
            # if this dataset belongs to training then shuffle the list of speakers
            random.shuffle(self.lst_spkr_np_files)

        else:
            self.utter_num = hp.m_ge2e.test_M
            self.min_utter_len = hp.m_ge2e.tt_data.min_test_utter_len

        self.training = training
        self.hp = hp

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        """
            this method is called everytime we fetch data using this dataset
            this method is called 'batch_size' number of times to produce that many data item
        :param idx: is the index of the speaker.
        :return: spectrogram utterance
        """

        # getting the file path
        spr_path = self.lst_spkr_np_files[idx]
        spr_path = os.path.join(self.data_path, spr_path)

        # loading the utterances
        spr_utters = np.load(spr_path)

        # get random idx to pull utterances
        utter_idx = np.random.randint(0, spr_utters.shape[0], self.utter_num)

        # getting random utterances
        spr_rnd_utters = spr_utters[utter_idx, :, :]

        # getting random clipping position
        random_clip = np.random.randint(0, spr_utters.shape[1] - self.min_utter_len - 1)

        # clipping the utterances
        spr_rnd_utters = spr_rnd_utters[:, random_clip:random_clip + self.min_utter_len, :]

        return spr_rnd_utters


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
                                   drop_last=True,
                                   prefetch_factor=2)

    test_dataset = EmbeddingModelTTDataset(hp=hp, training=False, data_path=test_specs_path)
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=hp.m_ge2e.test_N,  # number of speakers
                                  shuffle=False,
                                  drop_last=True,
                                  prefetch_factor=2)

    return train_loader, test_loader


# quick test, below code will not be executed when the file is imported
# it runs only when this file is directly executed
if __name__ == '__main__':
    pass
    #
    # from strings.constants import hp
    # import os
    #
    # hp.m_ge2e.training_N = 2
    #
    # trian_dl, test_dl = get_train_test_data_loader(hp)
    #
    # # this will produce 16 items because there are 17 speakers in the training and last is dropped
    #
    # itr_cnt = 1
    #
    # for j in range(itr_cnt):
    #     print("\nFresh batch")
    #     for i, mel_db_batch in enumerate(trian_dl):
    #         print(i, mel_db_batch.shape)
    #
    #
    # print("\n##### test data loader####")
    # # this will produce 4 items because there are 6 speakers in the test and last is dropped
    # for j in range(itr_cnt):
    #     for i, mel_db_batch in enumerate(test_dl):
    #         print(i, mel_db_batch.shape)
    #
    # print(1)
