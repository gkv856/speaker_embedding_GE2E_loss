from pathlib import Path

import torch

from utils.dict_to_dot import GetDictWithDotNotation

# project name
PROJECT_NAME = "AutoVoiceConversion"

current_dir = Path(__file__)
PROJECT_DIR = [p for p in current_dir.parents if p.parts[-1] == PROJECT_NAME][0]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device type available {device}")

hparam_dict = {

    # genereal parameters
    "general": {
        # small error
        "small_err": 1e-6,

        "is_training_mode": True,
        "device": device,
        "project_root": PROJECT_DIR,

    },

    # path to the raw audio file
    "raw_audio": {
        "raw_audio_path": "static/raw_data/wavs",
        "train_spectrogram_path": "static/spectrograms/train",
        "test_spectrogram_path": "static/spectrograms/test",
        "train_percent": .8,
    },

    ## Mel-filterbank
    "mel_fb": {
        "mel_window_length": 25,  # In milliseconds
        "mel_window_step": 10,  # In milliseconds
        "mel_n_channels": 80,
    },

    ## Audio
    "audio": {
        "sampling_rate": 16000,
        # Number of spectrogram frames in a partial utterance
        "partials_n_frames": 180,  # 1600 ms
    },

    ## Voice Activation Detection
    "vad": {
        # Window size of the VAD. Must be either 10, 20 or 30 milliseconds.
        # This sets the granularity of the VAD. Should not need to be changed.
        "vad_window_length": 30,  # In milliseconds
        # Number of frames to average together when performing the moving average smoothing.
        # The larger this value, the larger the VAD variations must be to not get smoothed out.
        "vad_moving_average_width": 8,
        # Maximum number of consecutive silent frames a segment can have.
        "vad_max_silence_length": 6,

        ## Audio volume normalization
        "audio_norm_target_dBFS": -30,

    },

    ## Generalized end2end Model loss parameters
    "m_ge2e": {
        "model_hidden_size": 256,
        "model_embedding_size": 256,
        "model_num_layers": 3,

        # setting the hyper parameters
        "lr": 0.05,
        "training_epochs": 1000,
        "model_path": "static/model_chk_pts/ge2e/ckpt_epoch_5000.pth",
        "restore_existing_model": False,
        "checkpoint_interval": 200,
        "training_N": 2,  # Number of  speaker aka batch_size for the data loader
        "training_M": 16,  # Number of utterances per speaker
        "test_N": 2,  # Number of  speaker aka batch_size for the data loader
        "test_M": 16,  # Number of utterances per speaker
    },

}

# this hp will be used throughout the project
hp = GetDictWithDotNotation(hparam_dict)

# adding some calculated hyper parameters
hp.audio.n_fft = int(hp.audio.sampling_rate * hp.mel_fb.mel_window_length / 1000)
hp.audio.hop_length = int(hp.audio.sampling_rate * hp.mel_fb.mel_window_step / 1000)
hp.vad.rate_partial_slices = 1.3
hp.vad.min_coverage = 0.75

# quick test, below code will not be executed when the file is imported
# it runs only when this file is directly executed
if __name__ == '__main__':
    print(hp)
