import torch.nn as nn
import torch
import os

class ModelGE2ELossSpeachEmbed(nn.Module):

    def __init__(self, hp):
        super(ModelGE2ELossSpeachEmbed, self).__init__()

        # this creates a three stacks (hp.num_layer) of LSTM
        self.LSTM_stack = nn.LSTM(hp.mel_fb.mel_n_channels,
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


def get_pre_trained_embedding_model(hp):
    """
    this method loads a pre-trained model and return the model in eval mode for inference/predictions
    :param hp:
    :return:
    """
    model = ModelGE2ELossSpeachEmbed(hp).to(hp.general.device)
    model_path = os.path.join(hp.general.project_root, hp.m_ge2e.best_model_path)

    # load weights as dictionary
    weight_dict = torch.load(model_path, map_location=hp.general.device)
    model.load_state_dict(weight_dict)
    model = model.eval()
    print(f"Pre-trained model loaded {model_path}")

    return model


# quick test, below code will not be executed when the file is imported
# it runs only when this file is directly executed
if __name__ == '__main__':
    from strings.constants import hp
    from s2_generalized_end2end_loss_GE2E.s1_dataset_loader import get_train_test_data_loader
    import os

    # testing the untrained model
    tmp_model = ModelGE2ELossSpeachEmbed(hp)

    train_specs_path = os.path.join(hp.general.project_root, hp.raw_audio.train_spectrogram_path)
    test_specs_path = os.path.join(hp.general.project_root, hp.raw_audio.test_spectrogram_path)

    train_dl, test_dl = get_train_test_data_loader(hp)

    total_utterances = hp.m_ge2e.training_N * hp.m_ge2e.training_M

    for mel_db_batch in train_dl:
        # mel is returned as 4x5x160x40 (batch x num_speakerxutterlenxn_mel)and we will reshape it to 20x160x40
        new_shape = (total_utterances, mel_db_batch.shape[2], mel_db_batch.shape[3])
        mel_db_batch = torch.reshape(mel_db_batch, new_shape)
        res = tmp_model(mel_db_batch)
        break

    assert res.shape[-1] == hp.m_ge2e.model_embedding_size
    print(res.shape, res)
