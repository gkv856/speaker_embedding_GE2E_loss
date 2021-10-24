import pickle
from tqdm import tqdm
import torch
import os

from wavenet_vocoder import builder
from tqdm import tqdm
import soundfile as sf

try:
    from s3_auto_voice_cloner.s4_auto_vc_network import get_pre_trained_auto_vc_network
    from utils.audio_utils import AudioUtils
    from utils.dict_to_dot import GetDictWithDotNotation

except:
    from AVC.s3_auto_voice_cloner.s4_auto_vc_network import get_pre_trained_auto_vc_network
    from AVC.utils.audio_utils import AudioUtils
    from AVC.utils.dict_to_dot import GetDictWithDotNotation


class VoiceCloner:

    def __init__(self, hp, tqdm, verbose=False):
        self.hp = hp
        self.verbose = verbose

        # loading pickle files created
        p1 = os.path.join(hp.general.project_root, hp.m_avc.s1.speaker_embs_metadata_path)
        speaker_embs_metadata_file = os.path.join(p1, hp.m_avc.s1.speaker_embs_metadata_file)

        # metadata contains npy file with each items having file path and speaker embeddings
        self.spkr_emb_md = pickle.load(open(speaker_embs_metadata_file, "rb"))

        # loading a pre-trained Auto Voice Clone model
        self.avc_model = get_pre_trained_auto_vc_network(hp)

        # loading audio utils
        self.au = AudioUtils(hp)

        # loading audio utils
        self.au = AudioUtils(hp)

        self.wavenet_model, self.w_hp = self.get_wave_net_model(hp)

        self.tqdm = tqdm

    def get_wave_net_model(self, pre_trained=True):
        # reading wavenet model's hyper parameters
        wave_net_hp = self.hp.m_wave_net.hp
        w_hp = GetDictWithDotNotation(wave_net_hp)

        wave_net_model = getattr(builder, w_hp.builder)(out_channels=w_hp.out_channels,
                                                        layers=w_hp.layers,
                                                        stacks=w_hp.stacks,
                                                        residual_channels=w_hp.residual_channels,
                                                        gate_channels=w_hp.gate_channels,
                                                        skip_out_channels=w_hp.skip_out_channels,
                                                        cin_channels=w_hp.cin_channels,
                                                        gin_channels=w_hp.gin_channels,
                                                        weight_normalization=w_hp.weight_normalization,
                                                        n_speakers=w_hp.n_speakers,
                                                        dropout=w_hp.dropout,
                                                        kernel_size=w_hp.kernel_size,
                                                        upsample_conditional_features=w_hp.upsample_conditional_features,
                                                        upsample_scales=w_hp.upsample_scales,
                                                        freq_axis_kernel_size=w_hp.freq_axis_kernel_size,
                                                        scalar_input=True,
                                                        legacy=w_hp.legacy,
                                                        )

        if pre_trained:
            m_path = os.path.join(self.hp.general.project_root, self.hp.m_wave_net.gen.best_model_path)
            checkpoint = torch.load(m_path, map_location=self.hp.general.device)
            wave_net_model.load_state_dict(checkpoint["state_dict"])

            wave_net_model = wave_net_model.eval().to(self.hp.general.device)

        return wave_net_model, w_hp

    def convert_mel_specs_to_audio(self, mel_specs=None):
        """
        This method converts the mel-spectrogram to an audio wav format using wavenet model
        :param wavenet_model:
        :param w_hp:
        :param c:
        :param tqdm:
        :return:
        """

        self.wavenet_model.eval()
        self.wavenet_model.make_generation_fast_()

        Tc = mel_specs.shape[0]
        upsample_factor = self.w_hp.hop_size

        # Overwrite length according to feature size
        length = Tc * upsample_factor

        # B x C x T
        c = torch.FloatTensor(mel_specs.T).unsqueeze(0)

        initial_input = torch.zeros(1, 1, 1).fill_(0.0)

        # Transform data to GPU
        initial_input = initial_input.to(self.hp.general.device)
        c = None if c is None else c.to(self.hp.general.device)

        with torch.no_grad():
            converted_wav = self.wavenet_model.incremental_forward(initial_input,
                                                              c=c,
                                                              g=None,
                                                              T=length,
                                                              tqdm=tqdm,
                                                              softmax=True,
                                                              quantize=True,
                                                              log_scale_min=self.w_hp.log_scale_min)

        converted_wav = converted_wav.view(-1).cpu().data.numpy()

        return converted_wav

    def create_cross_spkr_mel_spects(self, id_from, id_to, audio_from, audio_to=None):
        audio_from = torch.tensor(audio_from).unsqueeze(0).to(self.hp.general.device).float()

        # speaker embeddings
        emb_from = self.spkr_emb_md[id_from]
        emb_to = self.spkr_emb_md[id_to]

        emb_from = torch.tensor(emb_from).unsqueeze(0).to(self.hp.general.device).float()
        emb_to = torch.tensor(emb_to).unsqueeze(0).to(self.hp.general.device).float()

        with torch.no_grad():
            _, x_identic_psnt, _ = self.avc_model(audio_from, emb_from, emb_to)

            uttr_target = x_identic_psnt[0, :, :].cpu().numpy()

        if self.verbose:
            print(f"Mel-spect cloned {id_from} -> {id_to}!!")

        return uttr_target


# quick test, below code will not be executed when the file is imported
# it runs only when this file is directly executed
if __name__ == '__main__':
    from strings.constants import hp

    hp.m_avc.gen.best_model_path = "static/model_chk_pts/autovc/AVC_final_1000.pth"
    vcs_obj = VoiceCloner(hp, tqdm)

    path_audio = "static/raw_data/wavs/p225/p225_003.wav"
    path_audio = os.path.join(hp.general.project_root, path_audio)
    spkr_p225_mel_spec = vcs_obj.au.get_mel_spects_from_audio(path_audio, partial_slices=False)

    path_audio = "static/raw_data/wavs/p226/p226_003.wav"
    path_audio = os.path.join(hp.general.project_root, path_audio)
    spkr_p226_mel_spec = vcs_obj.au.get_mel_spects_from_audio(path_audio, partial_slices=False)

    avc_mel_specs = vcs_obj.create_cross_spkr_mel_spects("p225", "p226", spkr_p225_mel_spec[:320, :])

    np_audio = vcs_obj.convert_mel_specs_to_audio(avc_mel_specs)

    sf.write('test.wav', np_audio, hp.audio.sampling_rate, 'PCM_24')


    print(1)
