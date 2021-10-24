import numpy as np
import torch.utils.data as data
import pickle
import os
import random


class AutoVCNetDataset(data.Dataset):

    def __init__(self, hp):
        self.hp = hp

        # reading pickle file with speaker mel-spec files and embeddings
        file_path = os.path.join(hp.general.project_root, hp.m_avc.s1.speaker_embs_metadata_path)
        full_file_path = os.path.join(file_path, hp.m_avc.s1.speaker_embs_metadata_file)
        self.embs_md = pickle.load(open(full_file_path, "rb"))

        self.lst_spkr = []
        for k, v in self.embs_md.items():
            self.lst_spkr.append(k)

        # shuffling the list
        random.shuffle(self.lst_spkr)

        # storing the length to be returned in the __len__ method
        self.spr_len = len(self.lst_spkr)

        # reading pickle file with speaker mel-spec files and embeddings
        file_path = os.path.join(hp.general.project_root, hp.m_avc.tt_data.train_spects_path)

        self.dict_spr_utter_file = {}
        for spkr in self.lst_spkr:
            fp = os.path.join(file_path, f"sv_{spkr}.npy")
            self.dict_spr_utter_file[spkr] = fp

    def __len__(self):
        return self.spr_len

    def __getitem__(self, idx):
        selected_spr = self.lst_spkr[idx]

        spr_utter_path = self.dict_spr_utter_file[selected_spr]
        spr_emb = self.embs_md[selected_spr]

        # load utterance spectrogram of selected speaker
        # shape = 320x180x80
        utters = np.load(spr_utter_path)

        # getting a random utterances idx
        rnd_idx = np.random.randint(0, utters.shape[0])
        #         print(f"getting utter number {rnd_idx}")
        # fetching that utterances
        utter_used = utters[rnd_idx, :, :]

        # getting random length to crop
        # cropping the utterance length to 'self.hp.m_avc.s2.mul_32_utter_len' (128)
        left = np.random.randint(utter_used.shape[0] - self.hp.m_avc.s2.mul_32_utter_len)
        utter_used = utter_used[left:left + self.hp.m_avc.s2.mul_32_utter_len, :]

        return utter_used, spr_emb, spr_utter_path


def get_auto_vc_data_loader(hp, batch_size=1):
    train_dataset = AutoVCNetDataset(hp)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,  # number of speakers
                                   shuffle=True,
                                   drop_last=True)
    return train_loader


# quick test, below code will not be executed when the file is imported
# it runs only when this file is directly executed
if __name__ == '__main__':
    pass
    #
    from strings.constants import hp

    train_loader = get_auto_vc_data_loader(hp, batch_size=6)

    for i, res in enumerate(train_loader):
        print(i, res[0].shape, res[1].shape, res[2])

    print(1)