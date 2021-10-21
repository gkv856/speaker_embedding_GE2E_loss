from tqdm import tqdm
from wavenet_vocoder import builder
import torch
from utils.dict_to_dot import GetDictWithDotNotation
import os


def get_wave_net_model(hp, pre_trained=True):
    # reading wavenet model's hyper parameters
    wave_net_hp = hp.m_wave_net.hp
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
        m_path = os.path.join(hp.general.project_root, hp.m_wave_net.gen.best_model_path)
        checkpoint = torch.load(m_path, map_location=hp.general.device)
        wave_net_model.load_state_dict(checkpoint["state_dict"])

        wave_net_model = wave_net_model.eval().to(hp.general.device)

    return wave_net_model, w_hp


def convert_mel_specs_to_audio(wavenet_model, w_hp, hp, mel_specs=None, tqdm=tqdm):
    """
    This method converts the mel-spectrogram to an audio wav format using wavenet model
    :param wavenet_model:
    :param w_hp:
    :param c:
    :param tqdm:
    :return:
    """


    wavenet_model.eval()
    wavenet_model.make_generation_fast_()

    Tc = mel_specs.shape[0]
    upsample_factor = w_hp.hop_size

    # Overwrite length according to feature size
    length = Tc * upsample_factor

    # B x C x T
    c = torch.FloatTensor(mel_specs.T).unsqueeze(0)

    initial_input = torch.zeros(1, 1, 1).fill_(0.0)

    # Transform data to GPU
    initial_input = initial_input.to(hp.general.device)
    c = None if c is None else c.to(hp.general.device)

    with torch.no_grad():
        converted_wav = wavenet_model.incremental_forward(initial_input,
                                                          c=c,
                                                          g=None,
                                                          T=length,
                                                          tqdm=tqdm,
                                                          softmax=True,
                                                          quantize=True,
                                                          log_scale_min=w_hp.log_scale_min)

    converted_wav = converted_wav.view(-1).cpu().data.numpy()

    return converted_wav


# quick test, below code will not be executed when the file is imported
# it runs only when this file is directly executed
if __name__ == '__main__':
    from strings.constants import hp
    import soundfile as sf
    import pickle

    # getting wavenet model, this will up sample the mel-spec to final audio
    wave_net_model, wave_net_hp = get_wave_net_model(hp)

    # reading the cross speaker mel-spectrograms
    p1 = os.path.join(hp.general.project_root, hp.m_avc.gen.cross_mel_specs_path)
    file_path = os.path.join(p1, hp.m_avc.gen.cross_mel_specs_file)
    vc_cross_mel_specs = pickle.load(open(file_path, 'rb'))

    # looping over each mel-spec and save it as an audio
    for vc_cross_mel_spec in vc_cross_mel_specs:
        name = vc_cross_mel_spec[0]
        mel_specs = vc_cross_mel_spec[1]
        print(name)
        waveform = convert_mel_specs_to_audio(wave_net_model, wave_net_hp, mel_specs=mel_specs)
        sf.write(name + '.wav', waveform, 16000, 'PCM_24')

    print(1)
