import os

import torch
import torch.nn as nn


class ModelGE2ELossSpeachEmbed(nn.Module):

    def __init__(self, hp):
        super(ModelGE2ELossSpeachEmbed, self).__init__()

        # this creates a three stacks (hp.num_layer) of LSTM
        self.LSTM_stack = nn.LSTM(hp.audio.mel_n_channels,
                                  hp.m_ge2e.model_hidden_size,
                                  num_layers=hp.m_ge2e.model_num_layers,
                                  batch_first=True)

        for name, param in self.LSTM_stack.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        # feed forward layer
        self.projection = nn.Linear(hp.m_ge2e.model_hidden_size, hp.m_ge2e.model_embedding_size)

    def forward(self, x):
        x, _ = self.LSTM_stack(x.float())  # (batch, frames, n_mels)
        # only use last frame
        x = x[:, x.size(1) - 1]
        x = self.projection(x.float())

        # The embedding vector (d-vector) is defined as the L2 normalization of the network output
        x = x / torch.norm(x, dim=1).unsqueeze(1)
        return x


def get_pre_trained_embedding_model(hp, use_path_as_absolute=False):
    """
    this method loads a pre-trained model and return the model in eval mode for inference/predictions

    :param hp:
    :param use_path_as_absolute: when set to 'True' the 'hp.m_ge2e.best_model_path' path is treated as an absolute path
    and when set to 'False' the 'hp.m_ge2e.best_model_path' is joined with project root path.
    :return: Pre-trained Embedding model ready for inference
    """

    # loading untrained (empty) model
    embdding_model = ModelGE2ELossSpeachEmbed(hp).to(hp.general.device)

    # loading the pre-trained model path
    if use_path_as_absolute:
        model_path = hp.m_ge2e.best_model_path
    else:
        model_path = os.path.join(hp.general.project_root, hp.m_ge2e.best_model_path)

    # load weights as dictionary
    weight_dict = torch.load(model_path, map_location=hp.general.device)

    # setting model with pre-trained weights
    embdding_model.load_state_dict(weight_dict)

    # setting model to inference/prediction/eval mode
    embdding_model = embdding_model.eval().to(hp.general.device)
    print(f"Pre-trained model loaded from '{model_path}'")

    return embdding_model


# quick test, below code will not be executed when the file is imported
# it runs only when this file is directly executed
if __name__ == '__main__':
    pass

    # from strings.constants import hp
    # from embedding_model_GE2E.s1_dataset_loader import get_train_test_data_loader
    # import os
    #
    # # testing the untrained model
    # tmp_model = ModelGE2ELossSpeachEmbed(hp)
    #
    # train_dl, test_dl = get_train_test_data_loader(hp)
    #
    # total_utterances = hp.m_ge2e.training_N * hp.m_ge2e.training_M
    #
    # for mel_db_batch in train_dl:
    #     # mel is returned as 4x5x160x40 (batch x num_speakerxutterlenxn_mel)and we will reshape it to 20x160x40
    #     new_shape = (total_utterances, mel_db_batch.shape[2], mel_db_batch.shape[3])
    #     mel_db_batch = torch.reshape(mel_db_batch, new_shape)
    #     res = tmp_model(mel_db_batch)
    #     break
    #
    # assert res.shape[-1] == hp.m_ge2e.model_embedding_size
    # # print(res.shape, res)
