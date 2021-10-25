from pathlib import Path

import torch

try:
    from utils.dict_to_dot import GetDictWithDotNotation

    PROJECT_NAME = "AutoVoiceConversion"

    current_dir = Path(__file__)
    PROJECT_DIR = [p for p in current_dir.parents if p.parts[-1] == PROJECT_NAME][0]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device type available = '{device}'")

except:
    from AVC.utils.dict_to_dot import GetDictWithDotNotation

    PROJECT_NAME = "AVC"

    current_dir = Path("/content/AVC/strings/constants.py")
    PROJECT_DIR = [p for p in current_dir.parents if p.parts[-1] == PROJECT_NAME][0]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device type available = '{device}'")


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
    },

    ## Audio
    # same audio settings to be used in the wavenet model to reconstruct the audio from mel-spectrogram
    "audio": {
        "sampling_rate": 16000,

        "n_fft": 1024,  # 1024 seems to work well
        "hop_length": 1024 // 8,  # n_fft/4 seems to work better

        "mel_window_length": 25,  # In milliseconds
        "mel_window_step": 10,  # In milliseconds
        "mel_n_channels": 80,

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
        "rate_partial_slices": 1.3,
        "min_coverage": 0.75,
    },

    ## Generalized end2end Model loss parameters
    "m_ge2e": {
        "tt_data": {
            "train_spects_path": "static/spectrograms/m_ge2e/train",
            "test_spects_path": "static/spectrograms/m_ge2e/test",
            "train_percent": .8,
            "min_test_utter_len": 128,

            # Number of spectrogram frames in a partial utterance
            "min_train_utter_len": 160,  # 1600 ms
        },
        "model_hidden_size": 256,
        "model_embedding_size": 256,
        "model_num_layers": 3,

        # setting the hyper parameters
        "lr": 0.05,
        "training_epochs": 1000,
        "best_model_path": "static/model_chk_pts/ge2e/embedding_model_GE2E_loss_epoch_1000_L_0.024.pth",
        "checkpoint_dir": "static/model_chk_pts/ge2e",
        "save_best_weights": True,
        "min_test_loss": 2.0,

        "restore_existing_model": False,
        "checkpoint_interval": 200,
        "training_N": 2,  # Number of  speaker aka batch_size for the data loader
        "training_M": 16,  # Number of utterances per speaker
        "test_N": 2,  # Number of  speaker aka batch_size for the data loader
        "test_M": 16,  # Number of utterances per speaker
    },

    ## Auto voice cloner model
    "m_avc": {
        "tt_data": {
            "train_spects_path": "static/spectrograms/m_avc/train",
            "test_spects_path": "static/spectrograms/m_avc/test",
        },
        "gen": {
            "best_model_path": "static/model_chk_pts/autovc/AVC_ckpt_epoch_600.pth",
            "cross_mel_specs_path": "static/pickle_files",
            "cross_mel_specs_file": "spkr_cross_mel_specs_file.pkl",

        },

        "s1": {
            "num_uttrs": 10,
            "speaker_embs_metadata_path": "static/pickle_files",
            "speaker_embs_metadata_file": "speaker_embs_metadata.pkl",

        },

        "s2": {
            "mul_32_utter_len": 128,
        },

        # encoder model parameters
        "m_enc": {
            "dim_neck": 16,
            "freq": 16,
            "out_dims": 512,
            "lstm_enc_stack": 2,
        },

        # decoder model parameters
        "m_dec": {
            "out_dims": 512,
            "lstm_dec_stack": 1,
            "out_dim_lstm": 1024,
            "lstm_out_stack": 3,
        },

        # postnet model parameters
        "m_pn": {
            "out_dims": 512,
        },

        # AutoVC model training parameters
        "tpm": {
            "lambda_cd": 1,
            "num_iters": 10000,
            "log_step": 100,
            "dot_print": 10,
            "lr": 0.001,
            "reduce_lr_interval": 300,
            "checkpoint_dir": "static/model_chk_pts/autovc",
            "data_batch_size": 1,
            "checkpoint_interval": 200,
            "best_model_path": "static/model_chk_pts/autovc/m_best_50_L_0.0001.pth",
            "save_best_weights": True,
            "min_test_loss": 2.0,
        },
    },

    "m_wave_net": {
        "gen": {
            "best_model_path": "static/model_chk_pts/wavenet_model/wavenet_pretrained_step001000000_ema.pth"
        },
        "hp": {
            # DO NOT CHANGE THESE HP
            'name': "wavenet_vocoder",

            # Convenient model builder
            'builder': "wavenet",

            # Input type:
            # 1. raw [-1, 1]
            # 2. mulaw [-1, 1]
            # 3. mulaw-quantize [0, mu]
            # If input_type is raw or mulaw, network assumes scalar input and
            # discretized mixture of logistic distributions output, otherwise one-hot
            # input and softmax output are assumed.
            # **NOTE**: if you change the one of the two parameters below, you need to
            # re-run preprocessing before training.
            'input_type': "raw",
            'quantize_channels': 65536,  # 65536 or 256

            # Audio: these 4 items to be same as used to create mel out of audio
            # 'sample_rate': 16000,
            # 'fft_size': 1024,
            # # shift can be specified by either hop_size or frame_shift_ms
            # 'hop_size': 256,
            # 'num_mels': 80,

            # this is only valid for mulaw is True
            'silence_threshold': 2,

            'fmin': 125,
            'fmax': 7600,
            'frame_shift_ms': None,
            'min_level_db': -100,
            'ref_level_db': 20,
            # whether to rescale waveform or not.
            # Let x is an input waveform, rescaled waveform y is given by:
            # y = x / np.abs(x).max() * rescaling_max
            'rescaling': True,
            'rescaling_max': 0.999,
            # mel-spectrogram is normalized to [0, 1] for each utterance and clipping may
            # happen depends on min_level_db and ref_level_db, causing clipping noise.
            # If False, assertion is added to ensure no clipping happens.o0
            'allow_clipping_in_normalization': True,

            # Mixture of logistic distributions:
            'log_scale_min': float(-32.23619130191664),

            # Model:
            # This should equal to `quantize_channels` if mu-law quantize enabled
            # otherwise num_mixture * 3 (pi, mean, log_scale)
            'out_channels': 10 * 3,
            'layers': 24,
            'stacks': 4,
            'residual_channels': 512,
            'gate_channels': 512,  # split into 2 gropus internally for gated activation
            'skip_out_channels': 256,
            'dropout': 1 - 0.95,
            'kernel_size': 3,
            # If True, apply weight normalization as same as DeepVoice3
            'weight_normalization': True,
            # Use legacy code or not. Default is True since we already provided a model
            # based on the legacy code that can generate high-quality audio.
            # Ref: https://github.com/r9y9/wavenet_vocoder/pull/73
            'legacy': True,

            # Local conditioning (set negative value to disable))
            'cin_channels': 80,
            # If True, use transposed convolutions to upsample conditional features,
            # otherwise repeat features to adjust time resolution
            'upsample_conditional_features': True,
            # should np.prod(upsample_scales) == hop_size
            'upsample_scales': [4, 4, 4, 4],
            # Freq axis kernel size for upsampling network
            'freq_axis_kernel_size': 3,

            # Global conditioning (set negative value to disable)
            # currently limited for speaker embedding
            # this should only be enabled for multi-speaker dataset
            'gin_channels': -1,  # i.e., speaker embedding dim
            'n_speakers': -1,

            # Data loader
            'pin_memory': True,
            'num_workers': 2,

            # train/test
            # test size can be specified as portion or num samples
            'test_size': 0.0441,  # 50 for CMU ARCTIC single speaker
            'test_num_samples': None,
            'random_state': 1234,

            # Loss

            # Training:
            'batch_size': 2,
            'adam_beta1': 0.9,
            'adam_beta2': 0.999,
            'adam_eps': 1e-8,
            'amsgrad': False,
            'initial_learning_rate': 1e-3,
            # see lrschedule.py for available lr_schedule
            'lr_schedule': "noam_learning_rate_decay",
            'lr_schedule_kwargs': {},  # {"anneal_rate": 0.5, "anneal_interval": 50000},
            'nepochs': 2000,
            'weight_decay': 0.0,
            'clip_thresh': -1,
            # max time steps can either be specified as sec or steps
            # if both are None, then full audio samples are used in a batch
            'max_time_sec': None,
            'max_time_steps': 8000,
            # Hold moving averaged parameters and use them for evaluation
            'exponential_moving_average': True,
            # averaged = decay * averaged + (1 - decay) * x
            'ema_decay': 0.9999,

            # Save
            # per-step intervals
            'checkpoint_interval': 10000,
            'train_eval_interval': 10000,
            # per-epoch interval
            'test_eval_epoch_interval': 5,
            'save_optimizer_state': True,

            # Eval:
        }

    }
}

# this hp will be used throughout the project
hp = GetDictWithDotNotation(hparam_dict)

# few calculated values from wavenet model
hp.m_wave_net.hp.sample_rate = hp.audio.sampling_rate
hp.m_wave_net.hp.fft_size = hp.audio.n_fft
hp.m_wave_net.hp.hop_size = hp.audio.hop_length
hp.m_wave_net.hp.num_mels = hp.audio.mel_n_channels

# adding some calculated hyper parameters

# quick test, below code will not be executed when the file is imported
# it runs only when this file is directly executed
if __name__ == '__main__':
    print(hp)
