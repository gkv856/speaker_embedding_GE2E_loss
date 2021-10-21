from s2_generalized_end2end_loss_GE2E.s1_dataset_loader import get_train_test_data_loader
import torch
import torch.autograd as grad
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from s2_generalized_end2end_loss_GE2E.s3_loss_function_GE2E import GE2ELoss


def calculate_ERR(model, hp, N=4, M=16):
    hp.m_ge2e.test_N = N
    hp.m_ge2e.test_M = M

    # creating train and test loaders
    _, test_loader = get_train_test_data_loader(hp=hp)

    # calculating the total sentences/utterances
    total_utterances = hp.m_ge2e.test_N * hp.m_ge2e.test_M

    # initializing w and b place holders
    w = grad.Variable(torch.tensor(1.0))
    b = grad.Variable(torch.tensor(0.0))

    for mel_db_batch in test_loader:
        # mel_db_batch is returned as 4x5x160x40 (batchxnum_speakerxutterlenxn_mel) and we will reshape it to 20x160x40
        new_shape = (total_utterances, mel_db_batch.size(2), mel_db_batch.size(3))
        mel_db_batch = torch.reshape(mel_db_batch, new_shape)

        # pass mel_db_batch through the pre-trained model
        embeddings = model(mel_db_batch)

        # output of the model is (NxM, utterance_size), we reshape it to NxMxUtter_size
        embeddings = torch.reshape(embeddings, (hp.m_ge2e.test_N, hp.m_ge2e.test_M, embeddings.size(1)))

        # calculating centroids and similarity matrix
        centroids = GE2ELoss.get_centroids(embeddings)
        cos_sim = GE2ELoss.get_cos_sim(embeddings, centroids, hp)
        sim_matrix = w * cos_sim + b

        S = sim_matrix.detach().numpy()
        print(S.shape, type(S), "\n", S)

        # calculating EER
        diff = 1
        EER = 0
        EER_thres = 0
        EER_FAR = 0
        EER_FRR = 0

        # through thresholds calculate false acceptance ratio (FAR) and false reject ratio (FRR)
        thres_lst = [0.01 * i + 0.5 for i in range(50)]
        for thres in thres_lst:
            S_thres = S > thres

            # False acceptance ratio = false acceptance / mismatched population (enroll speaker != verification speaker)
            # sum of number of times there were a TRUE in the fist axis of the matrix
            # sum of number of times there was a TRUE for the same person
            # e.g.
            # t[0] =
            # array([[ True, False, False, False],
            #        [ True, False, False, False],
            #        [ True, False, False, False],
            #        [ True, False, False, False],
            #        [ True, False, False, False],
            #        [ True, False, False, False]])
            # hence np.sum(S_thres[i]) = 6

            # and t[0, :, 0]
            # array([ True,  True,  True,  True,  True,  True])
            # hence np.sum(S_thres[i, :, i] = 6
            # hence 6 - 6 = 0 for i = 0
            # and [0, 0, 0, 0] for all the speakers
            # which means false acceptance ratio was 0.. which is awesome

            denominator = (hp.m_ge2e.test_N - 1) / hp.m_ge2e.test_M / hp.m_ge2e.test_N
            lst = [np.sum(S_thres[i]) - np.sum(S_thres[i, :, i]) for i in range(hp.m_ge2e.test_N)]
            FAR = sum(lst) / denominator

            # False reject ratio = false reject / matched population (enroll speaker = verification speaker)
            # this same but in reverse order.
            # we have M number of speakers and for given number, how many did we reject?
            denominator = hp.m_ge2e.test_M / hp.m_ge2e.test_N
            lst = [hp.m_ge2e.test_M - np.sum(S_thres[i][:, i]) for i in range(hp.m_ge2e.test_N)]
            FRR = sum(lst) / denominator

            # Save threshold when FAR = FRR (=EER)
            if diff > abs(FAR - FRR):
                diff = abs(FAR - FRR)
                EER = (FAR + FRR) / 2
                EER_thres = thres
                EER_FAR = FAR
                EER_FRR = FRR

        print("\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EER, EER_thres, EER_FAR, EER_FRR))


def plot_scatter(model, hp, N=4, M=16):
    hp.m_ge2e.test_N = N
    hp.m_ge2e.test_M = M

    labels = []
    for i in range(hp.m_ge2e.test_N):
        speaker_num = f"S{i + 1}"
        labels.extend([speaker_num] * hp.m_ge2e.test_M)

    # print(labels)

    # creating train and test loaders
    _, test_loader = get_train_test_data_loader(hp=hp)

    # calculating the total sentences/utterances
    total_utterances = hp.m_ge2e.test_N * hp.m_ge2e.test_M

    # initializing w and b place holders
    w = grad.Variable(torch.tensor(1.0))
    b = grad.Variable(torch.tensor(0.0))

    for mel_db_batch in test_loader:
        # mel_db_batch is returned as 4x5x160x40 (batchxnum_speakerxutterlenxn_mel) and we will reshape it to 20x160x40
        new_shape = (total_utterances, mel_db_batch.size(2), mel_db_batch.size(3))
        mel_db_batch = torch.reshape(mel_db_batch, new_shape)

        # pass mel_db_batch through the pre-trained model
        embeddings = model(mel_db_batch)

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


# quick test, below code will not be executed when the file is imported
# it runs only when this file is directly executed
if __name__ == '__main__':
    from strings.constants import hp
    from s2_generalized_end2end_loss_GE2E.s2_model_GE2E_loss_speach_embed import  get_pre_trained_embedding_model

    # loading a pre-trained model
    model = get_pre_trained_embedding_model(hp)

    # calculating ERR
    calculate_ERR(model, hp, 4, 8)

    # plotting speaker embeddings
    plot_scatter(model, hp, 4, 16)

    print(1)
