import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os

try:
    from s3_auto_voice_cloner.s2_auto_vc_dataloader import get_auto_vc_data_loader
    from s3_auto_voice_cloner.s3_auto_vc_models import EncoderModel, DecoderModel, Postnet
except:
    from AVC.s3_auto_voice_cloner.s2_auto_vc_dataloader import get_auto_vc_data_loader
    from AVC.s3_auto_voice_cloner.s3_auto_vc_models import EncoderModel, DecoderModel, Postnet


class AutoVCNetwork(nn.Module):
    """
        AutoVCNetwork combines all the three models/networks defined above. The training and loss will be run on this network
    """

    def __init__(self, hp):
        super(AutoVCNetwork, self).__init__()

        self.hp = hp
        self.encoder = EncoderModel(hp)
        self.decoder = DecoderModel(hp)
        self.postnet = Postnet(hp)

    def forward(self, mel_specs, original_emb, target_emb):

        # passing mel_specs and original embedding for the speaker through encoder model
        # this will return codes with dimensions
        # len(codes), codes[0].shape = (8, torch.Size([2, 32]))
        # this is up1 and up2 in figure 3(c)

        # encoder's job is take mel spec and give out data that is pure content, no speaker related info
        spkr_content_wo_embs = self.encoder(mel_specs, original_emb)
        if target_emb is None:
            return torch.cat(spkr_content_wo_embs, dim=-1)

        tmp = []
        for code in spkr_content_wo_embs:
            tmp.append(code.unsqueeze(1).expand(-1, int(mel_specs.shape[1] / len(spkr_content_wo_embs)), -1))
        spkr_content_wo_embs_exp = torch.cat(tmp, dim=1)

        # concatenating the up1, up2 and original embedding (target_emb)
        # in the fig 3(c) this is equivalent to concatenate layer of 320 length
        # target embs is a speaker's style of spkeaking. For training, to and from are same but during conversion these
        # will be different
        tgt_embs = target_emb.unsqueeze(1).expand(-1, mel_specs.shape[1], -1)

        # joining speaker content and speaker's speaking style (speaker emb)
        encoder_outputs = torch.cat((spkr_content_wo_embs_exp, tgt_embs), dim=-1)

        # mel_outputs is the output of '1×1 Conv' layer in fig 3(c)
        # mel_outputs =  torch.Size([2, 128, 80])
        ypred_mel_spects = self.decoder(encoder_outputs)

        # mel_outputs_postnet is the output of '5×1 ConvNorm' without addition
        # mel_outputs_postnet.shape = torch.Size([2, 80, 128])
        residual_mel_spect_pn = self.postnet(ypred_mel_spects.transpose(2, 1))

        # this is the final re-created mel_specs output
        # ideally this should be very close to the starting mel_specs.
        # this will be used to calculate the loss
        ypred_mel_spects_final = ypred_mel_spects + residual_mel_spect_pn.transpose(2, 1)

        # finally return all the outputs to calculate losses
        # (torch.Size([2, 128, 80]), torch.Size([2, 128, 80]), torch.Size([2, 256]))
        ypred_spkr_content = torch.cat(spkr_content_wo_embs, dim=-1)

        return ypred_mel_spects, ypred_mel_spects_final, ypred_spkr_content


def get_pre_trained_auto_vc_network(hp, absolute_path=False):
    # creating an instance of the AutoVC network
    # sending the model to the GPU (if available)
    auto_vc_net = AutoVCNetwork(hp).to(hp.general.device)

    if absolute_path:
        model_path = hp.m_avc.gen.best_model_path
    else:
        model_path = os.path.join(hp.general.project_root, hp.m_avc.gen.best_model_path)

    # load weights as dictionary
    weight_dict = torch.load(model_path, map_location=hp.general.device)
    auto_vc_net.load_state_dict(weight_dict)
    auto_vc_net = auto_vc_net.eval().to(hp.general.device)
    print(f"Pre-trained model loaded from '{model_path}'")

    return auto_vc_net


# quick test, below code will not be executed when the file is imported
# it runs only when this file is directly executed
if __name__ == '__main__':
    from strings.constants import hp

    auto_vc_net = AutoVCNetwork(hp)
    # print(auto_vc_net)

    train_dl = get_auto_vc_data_loader(hp, batch_size=1)
    for i, res in enumerate(train_dl):
        # specs = res[0]  # torch.tensor(res[0]).float()
        # emb = res[1]  # torch.tensor(res[1]).float()

        specs = torch.tensor(res[0]).float()
        emb = torch.tensor(res[1]).float()
        # print(emb.shape, specs.shape)

        mel_outputs, mel_outputs_postnet, generated_embs = auto_vc_net(specs, emb, emb)
        # print("res = ", mel_outputs.shape, mel_outputs_postnet.shape, generated_embs.shape)

        assert mel_outputs.shape[1] == hp.m_avc.s2.mul_32_utter_len
        assert mel_outputs.shape[2] == hp.audio.mel_n_channels

        assert mel_outputs_postnet.shape[1] == hp.m_avc.s2.mul_32_utter_len
        assert mel_outputs_postnet.shape[2] == hp.audio.mel_n_channels

        assert generated_embs.shape[1] == hp.m_ge2e.model_embedding_size

    print("Success, all test passed!!")
