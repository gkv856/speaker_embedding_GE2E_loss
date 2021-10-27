import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# creating a linear normalized layer
class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        # initializing the conv weights as xavier uniform distribution
        gain = torch.nn.init.calculate_gain(w_init_gain)
        torch.nn.init.xavier_uniform_(self.linear_layer.weight, gain=gain)

    def forward(self, x):
        return self.linear_layer(x)


# creating a convolutional->normalization layer

class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None,
                 dilation=1, bias=True, w_init_gain='linear', norm_batch=True):
        super(ConvNorm, self).__init__()

        self.norm_batch = norm_batch
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        # conv layer
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    bias=bias)

        # normalization layer
        self.bn = nn.BatchNorm1d(out_channels)

        # initializing the conv weights as xavier uniform distribution
        gain = torch.nn.init.calculate_gain(w_init_gain)
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=gain)

    def forward(self, signal):
        conv_signal = self.conv(signal)
        if self.norm_batch:
            conv_signal = self.bn(conv_signal)

        return conv_signal


class EncoderModel(nn.Module):
    """
        https://arxiv.org/abs/1905.05879

        Encoder module: this is the model defined in the paper in section 3(a) on page 6
        Note that the Es(.) in the 3(a) is nothing but the embeddings.
        We concatenate the embeddings in the forward method below hence the input dimensions for the
        first conv layer is kwargs["n_mels"] + kwargs["dim_emb"]

        the output is the up1 and up2 mentioned in the section 3(c)

    """

    def __init__(self, hp):
        super(EncoderModel, self).__init__()

        self.hp = hp

        convolutions = []
        for i in range(3):
            # input dimensions = n_mels + embedding dims for the first layer
            # (bkz we will concat embeds and mels specs)
            # for subsequent layers it will = out_dims
            # in_dims = hp.audio.mel_n_channels + hp.m_ge2e.model_embedding_size if i == 0 else hp.m_avc.m_enc.out_dims
            in_dims = hp.audio.mel_n_channels + hp.m_ge2e.model_embedding_size if i == 0 else hp.m_avc.m_enc.out_dims

            conv_layer = ConvNorm(in_dims,
                                  hp.m_avc.m_enc.out_dims,
                                  kernel_size=5,
                                  stride=1,
                                  padding=2,
                                  dilation=1,
                                  w_init_gain='relu',
                                  norm_batch=hp.m_avc.tpm.norm_batch)

            convolutions.append(conv_layer)

        # storing the layers into ModuleList
        self.convolutions = nn.ModuleList(convolutions)

        # creating the LSTM stacked layer and with bi-directional=true
        # the output of convnorm will be input for the LSTM
        # dimensions neck will be 32 = 16x2 since we have stacked LSTM
        self.lstm = nn.LSTM(hp.m_avc.m_enc.out_dims,
                            hp.m_avc.m_enc.dim_neck,
                            hp.m_avc.m_enc.lstm_enc_stack,
                            batch_first=True,
                            bidirectional=True)

    def forward(self, mel_spec, emb_original):
        """
            mel_spec = num_utterance x utterance_length x n_mels (2x160x80)
            emb_original = num_utterance x embedding_size (2x256)
        """
        # reshaping it to num_utterance x n_mels x utterance_length (2x80x160)
        x = mel_spec.squeeze(1).transpose(2, 1)

        # to concatenate mel_spec and embedding, we need to make them of same dimensions
        # unsqueeze will increase a dimension and expand will copy the same values to each dimensions
        c_org = emb_original.unsqueeze(-1).expand(-1, -1, x.size(-1))

        x = torch.cat((x, c_org), dim=1)

        for conv in self.convolutions:
            x = F.relu(conv(x))

        # changing x from 2x512x128 to 2x128x512
        x = x.transpose(1, 2)

        # flatten parameters to bring everything to a single memory stack, nothing related to model.
        self.lstm.flatten_parameters()

        # outputs.shape = torch.Size([2, 128, 32])
        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :self.hp.m_avc.m_enc.dim_neck]
        out_backward = outputs[:, :, self.hp.m_avc.m_enc.dim_neck:]

        codes = []
        for i in range(0, outputs.shape[1], self.hp.m_avc.m_enc.freq):
            con_cat = torch.cat((out_forward[:, i + self.hp.m_avc.m_enc.freq - 1, :], out_backward[:, i, :]), dim=-1)
            codes.append(con_cat)

        # len(codes) = 8
        # codes[0].shape = torch.Size([2, 32])
        return codes


class DecoderModel(nn.Module):
    """
        https://arxiv.org/abs/1905.05879

        Decoder module:: this is half the model defined in the paper in section 3(c) on page 6 until 1x1 conv

        the output is the of torch.Size([2, 128, 80])
    """

    def __init__(self, hp):
        super(DecoderModel, self).__init__()

        self.hp = hp

        in_dims = hp.m_avc.m_enc.dim_neck * 2 + hp.m_ge2e.model_embedding_size

        self.lstm_concat = nn.LSTM(in_dims, hp.m_avc.m_dec.out_dims, hp.m_avc.m_dec.lstm_dec_stack, batch_first=True)

        convolutions = []
        for i in range(3):
            conv_layer = ConvNorm(hp.m_avc.m_dec.out_dims,
                                  hp.m_avc.m_dec.out_dims,
                                  kernel_size=5,
                                  stride=1,
                                  padding=2,
                                  dilation=1,
                                  w_init_gain='relu',
                                  norm_batch=hp.m_avc.tpm.norm_batch)

            convolutions.append(conv_layer)

        # storing the layers into ModuleList
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm_out = nn.LSTM(hp.m_avc.m_dec.out_dims,
                                hp.m_avc.m_dec.out_dim_lstm,
                                hp.m_avc.m_dec.lstm_out_stack,
                                batch_first=True)

        self.linear_projection = LinearNorm(hp.m_avc.m_dec.out_dim_lstm, hp.audio.mel_n_channels)

    def forward(self, x):
        """
            x.shape = torch.Size([2, 128, 288])
        """
        # self.lstm1.flatten_parameters()
        x, _ = self.lstm_concat(x)
        x = x.transpose(1, 2)

        for conv in self.convolutions:
            x = F.relu(conv(x))

        x = x.transpose(1, 2)

        outputs, _ = self.lstm_out(x)

        decoder_output = self.linear_projection(outputs)

        return decoder_output


class Postnet(nn.Module):
    """
        https://arxiv.org/abs/1905.05879

        Postnet module:: this is second half of  the model defined in the paper in section 3(c) on page 6
        after 1x1 conv

        mel_outputs_postnet.shape = torch.Size([2, 80, 128])
        this then will be sum/add to the 'mel_outputs' of the decoder model to get the final mel_specs

        this mel_specs must be same as original mel_specs (this is our loss that the network needs to minimize)

        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hp):
        super(Postnet, self).__init__()

        self.hp = hp

        self.convolutions = nn.ModuleList()

        # creating 1st conv layer with different input and out
        self.convolutions.append(ConvNorm(hp.audio.mel_n_channels,
                                          hp.m_avc.m_pn.out_dims,
                                          kernel_size=5,
                                          stride=1,
                                          padding=2,
                                          dilation=1,
                                          w_init_gain='tanh',
                                          norm_batch=hp.m_avc.tpm.norm_batch)
                                 )

        # adding 4 more conv batch norm layers
        for i in range(1, 5 - 1):
            self.convolutions.append(ConvNorm(hp.m_avc.m_pn.out_dims,
                                              hp.m_avc.m_pn.out_dims,
                                              kernel_size=5,
                                              stride=1,
                                              padding=2,
                                              dilation=1,
                                              w_init_gain='tanh',
                                              norm_batch=hp.m_avc.tpm.norm_batch)
                                     )

        # adding the final output conv layer
        self.convolutions.append(ConvNorm(hp.m_avc.m_pn.out_dims,
                                          hp.audio.mel_n_channels,
                                          kernel_size=5,
                                          stride=1,
                                          padding=2,
                                          dilation=1,
                                          w_init_gain='linear',
                                          norm_batch=hp.m_avc.tpm.norm_batch)
                                 )

    def forward(self, x):
        """
            input shape = x.shape = mel_outputs.transpose(2, 1).shape = torch.Size([2, 80, 128])
        """
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)

        return x


# quick test, below code will not be executed when the file is imported
# it runs only when this file is directly executed
if __name__ == '__main__':
    from strings.constants import hp

    enc_model = EncoderModel(hp)
    # print(enc_model)

    # creating some random inputs to test the model
    batch_size = 2
    specs = torch.tensor(np.random.rand(batch_size, hp.m_avc.s2.mul_32_utter_len, hp.audio.mel_n_channels)).float()
    emb = torch.tensor(np.random.rand(batch_size, hp.m_ge2e.model_embedding_size)).float()

    codes = enc_model(specs, emb)
    assert codes[0].shape[1] == hp.m_avc.m_enc.dim_neck * 2
    # print(len(codes), codes[0].shape)

    # #######################################
    # # testing decoder model
    #
    # # testing the model with random values

    dec_model = DecoderModel(hp)
    # print(dec_model)
    # creating some random inputs to test the model
    in_dims = hp.m_avc.m_enc.dim_neck * 2 + hp.m_ge2e.model_embedding_size
    x = torch.tensor(np.random.rand(batch_size, hp.m_avc.s2.mul_32_utter_len, in_dims)).float()

    res = dec_model(x)
    assert res.shape[-1] == hp.audio.mel_n_channels
    # print(x.shape, res.shape)

    # #############################
    # # testing postnet model

    # testing the model with random values

    pn_model = Postnet(hp)
    # print(pn_model)

    # creating some random inputs to test the model
    x = torch.tensor(np.random.rand(batch_size, hp.audio.mel_n_channels, hp.m_avc.s2.mul_32_utter_len)).float()

    res = pn_model(x)
    # print(x.shape, res.shape)
    assert res.shape[-1] == hp.m_avc.s2.mul_32_utter_len

    print("Success, all test passed!!")
