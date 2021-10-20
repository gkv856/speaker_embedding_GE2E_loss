import os
import numpy as np

from utils.audio_utils import preprocess_wav, compute_partial_slices, wav_to_mel_spectrogram, shuffle_along_axis


def save_spectrogram_tisv(hp, speaker_utter_cnt=100):
    """ Full preprocess of text independent utterance. The log-mel-spectrogram is saved as numpy file.
        Each partial utterance is split by voice detection using DB
        and the first and the last 180 frames from each partial utterance are saved.
        Need : utterance data set (VTCK)
    """
    print("Text independent speaker verification (TISV) utterance feature extraction started..")

    # Create folder if does not exist, if exist then ignore
    spec_path_train = os.path.join(hp.general.project_root, hp.raw_audio.train_spectrogram_path)
    spec_path_test = os.path.join(hp.general.project_root, hp.raw_audio.test_spectrogram_path)
    os.makedirs(spec_path_train, exist_ok=True)
    os.makedirs(spec_path_test, exist_ok=True)

    # list of folders (speakers) in the folder
    audio_path = os.path.join(hp.general.project_root, hp.raw_audio.raw_audio_path)
    lst_all_speaker_folders = os.listdir(audio_path)
    total_speaker_num = len(lst_all_speaker_folders)

    print(f"Total speakers to be saved {total_speaker_num}")
    # looping through each speaker
    for i, folder in enumerate(lst_all_speaker_folders):

        # path of each speaker

        per_speaker_folder = os.path.join(audio_path, folder)
        per_speaker_wavs = os.listdir(per_speaker_folder)

        print(f"\nProcessing speaker no. {i + 1} with '{len(per_speaker_wavs)}'' audio files")
        utterances_spec = []

        # looping through all the folders for a given speaker
        for utter_wav_file in per_speaker_wavs:
            # path of each utterance
            utter_wav_file_path = os.path.join(per_speaker_folder, utter_wav_file)

            # if utter len already more than requested len, then no need to check other audio files
            # meaning for a given speaker we already have 'speaker_utter_cnt' number of audio samples
            if type(utterances_spec) == np.ndarray and utterances_spec.shape[0] >= speaker_utter_cnt:
                # print(f"Breaking the loop, speaker utter count already achieved")
                break

            wav = preprocess_wav(utter_wav_file_path, hp=hp)

            wav_slices, mel_slices = compute_partial_slices(len(wav), hp=hp)
            max_wave_length = wav_slices[-1].stop
            if max_wave_length >= len(wav):
                wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

            # Split the utterance into partials and forward them through the model
            mel = wav_to_mel_spectrogram(wav, hp)
            mels = np.array([mel[s] for s in mel_slices])

            if not type(utterances_spec) == np.ndarray:
                utterances_spec = mels
            else:
                utterances_spec = np.concatenate((utterances_spec, mels), axis=0)

        # collecting all the utterances across all the wav files into np array
        # Checking if speaker's utterance qualifies to be used. i.e. a min utterance length is available in the audio
        if utterances_spec.shape[0] >= speaker_utter_cnt:
            shuffled_utter_specs = shuffle_along_axis(utterances_spec, axis=0)
            train_idx = int(speaker_utter_cnt * hp.raw_audio.train_percent)

            train_data = shuffled_utter_specs[:train_idx, :, :]
            test_data = shuffled_utter_specs[train_idx:, :, :]

            print(f"\n'Training data' Size saved = {train_data.shape}\n")
            file_full_path = os.path.join(spec_path_train, f"sv_{folder}_{i + 1}.npy")
            np.save(file_full_path, train_data)

            print(f"\n'Eval data' Size saved = {test_data.shape}\n")
            file_full_path = os.path.join(spec_path_test, f"sv_{folder}_{i + 1}.npy")
            np.save(file_full_path, test_data)

        else:
            print(f"\nSkipped for {folder} and Size was = {utterances_spec.shape}\n")

    print("Spectrograms saved!!")


# quick test, below code will not be executed when the file is imported
# it runs only when this file is directly executed
if __name__ == '__main__':
    from strings.constants import hp
    save_spectrogram_tisv(hp, 2)
