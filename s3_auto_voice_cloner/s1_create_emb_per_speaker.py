import numpy as np
import torch
import os
import pickle

from s2_generalized_end2end_loss_GE2E.s2_model_GE2E_loss_speach_embed import get_pre_trained_embedding_model


def create_embbedings_per_speaker(hp):
    # getting all the numpy files in tmp list
    utterances = []

    # data path saved
    train_specs_path = os.path.join(hp.general.project_root, hp.m_avc.tt_data.train_spects_path)

    # getting all the folders/speakers
    data_iter = iter(os.walk(train_specs_path))

    # note: walk returns a tuple
    # 0th item -> name of current folder
    # 1st item -> list of folders in the current folder
    # 2rd item -> files in current folder

    iter_data = next(data_iter)
    lst_spkr_np_files = iter_data[2]

    # storing the numpy file full-path in lst_np_file
    lst_np_file = [os.path.join(train_specs_path, x) for x in lst_spkr_np_files]

    # loading a pre-trained embedding model
    emb_model = get_pre_trained_embedding_model(hp)

    for spr in lst_np_file:

        # loading a speaker's utterance in mel-spec format
        utters = np.load(spr)

        u_st = 0
        u_ed = 1
        embs = []
        for i in range(hp.m_avc.s1.num_uttrs):
            # getting speaker's voice
            utter = utters[u_st:u_ed, :, :]
            u_st = u_ed
            u_ed += 1

            # creating tensor
            utter_t = torch.tensor(utter).to(hp.general.device)

            # getting embedding for the utterance
            emb = emb_model(utter_t)

            # putting embeddings into a list
            embs.append(emb.detach().squeeze().cpu().numpy())
            # print(utter_t.shape, embs[0].shape)

        # taking average of all the embeddings of a speaker
        embs_mean = np.mean(embs, axis=0)

        # saving speaker np file path and embs
        utterances.append((spr, embs_mean))

    file_path = os.path.join(hp.general.project_root, hp.m_avc.s1.speaker_embs_metadata_path)
    full_file_path = os.path.join(file_path, hp.m_avc.s1.speaker_embs_metadata_file)
    # saving utterances as a train.pkl file
    with open(full_file_path, 'wb') as handle:
        pickle.dump(utterances, handle)

    print("File saved!!")

    return utterances


# quick test, below code will not be executed when the file is imported
# it runs only when this file is directly executed
if __name__ == '__main__':
    pass

    from strings.constants import hp
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    hp.m_ge2e.best_model_path = "static/model_chk_pts/ge2e/final_epoch_1000_L_0.0390.pth"
    utterances = create_embbedings_per_speaker(hp)

    labels = ["s1", "s2", "s3", "s4"]

    embs = []
    for u in utterances:
        embs.append(u[1])

    embeddings = torch.tensor(embs)

    scatters = TSNE(n_components=2, random_state=0).fit_transform(embeddings.cpu().detach().numpy())
    fig = plt.figure(figsize=(5, 5))

    current_Label = labels[0]
    current_Index = 0
    for index, label in enumerate(labels[1:], 1):
        if label != current_Label:
            plt.scatter(scatters[current_Index:index, 0], scatters[current_Index:index, 1],
                        label='{}'.format(current_Label))
            current_Label = label
            current_Index = index
    plt.scatter(scatters[current_Index:, 0], scatters[current_Index:, 1], label='{}'.format(current_Label))
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(1)
