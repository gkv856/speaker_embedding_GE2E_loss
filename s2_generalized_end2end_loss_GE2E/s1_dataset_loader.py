import random

import numpy as np
import torch.utils.data as data
import os

class TrainTestDataset(data.Dataset):
    """
        this class will be used to fetch the spectrogram prepared in the step1
        these spectrograms will be used to train speaker verification model using GE2E loss

    """

    def __init__(self, data_path, hp, training=True):

        # getting all the numpy files in tmp list
        tmp = [x for x in os.walk(data_path)][0][-1]

        # storing the numpy file full-path in lst_np_file
        tmp_path = os.path.join(hp.general.project_root, hp.raw_audio.train_spectrogram_path)
        self.lst_np_file = [os.path.join(tmp_path, x) for x in tmp]

        # getting total speakers
        self.data_len = len(self.lst_np_file)

        if training:
            self.utter_num = hp.m_ge2e.training_M

        else:
            self.utter_num = hp.m_ge2e.test_M

        self.training = training

        # if this dataset belongs to training then shuffle the list of speakers
        if self.training:
            random.shuffle(self.lst_np_file)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        """
            this method is called everytime we fetch data using this dataset
            this method is called 'batch_size' number of times to produce that many data item
        :param idx: is the index of the speaker.
        :return: spectrogram utterance
        """

        selected_file = self.lst_np_file[idx]

        # load utterance spectrogram of selected speaker
        utters = np.load(selected_file)

        # randomly select M utterances per speaker
        utter_index = np.random.randint(0, utters.shape[0], self.utter_num)
        utterance = utters[utter_index]

        return utterance


def get_train_test_data_loader(hp):
    """
        creates data loader for training and testing
    :param hp:
    :param lst_train_data: list of numpy files (np array of spectrograms) created in Step1
    :param lst_test_data: list of numpy files (np array of spectrograms) created in Step1
    :return:
    """

    train_specs_path = os.path.join(hp.general.project_root, hp.raw_audio.train_spectrogram_path)
    test_specs_path = os.path.join(hp.general.project_root, hp.raw_audio.test_spectrogram_path)

    train_dataset = TrainTestDataset(hp=hp, training=True, data_path=train_specs_path)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=hp.m_ge2e.training_N,  # number of speakers
                                   shuffle=True,
                                   drop_last=True)

    # TODO implement num_workers
    # num_workers=hp.m_ge2e.training_num_workers,

    test_dataset = TrainTestDataset(hp=hp, training=False, data_path=test_specs_path)
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
