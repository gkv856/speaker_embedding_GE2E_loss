import pickle
import os
import numpy as np
import torch
import random

from s3_auto_voice_cloner.s4_auto_vc_network import get_pre_trained_auto_vc_network


def create_mel_specs_per_speaker(hp):
    """
    this method reads the pickle file created in step1 of Auto VC model.
    this metadata is further used to create results pickle file that contains voice conversion per speaker
    i.e S1->S2, S1->S3,.... S1->SN
    :param hp:
    :return:
    """
    # loading pickle files created
    p1 = os.path.join(hp.general.project_root, hp.m_avc.s1.speaker_embs_metadata_path)
    speaker_embs_metadata_file = os.path.join(p1, hp.m_avc.s1.speaker_embs_metadata_file)

    # metadata contains npy file with each items having file path and speaker embeddings
    metadata = pickle.load(open(speaker_embs_metadata_file, "rb"))

    # loading a pre-trained Auto Voice Clone model
    auto_vc_model = get_pre_trained_auto_vc_network(hp)

    # all the cross spects will be stored here
    voice_cloned_spects = []

    for spkr_i in metadata:

        # speaker name depends on how the file name is structured therefore needs to split accordingly
        spkr_i_name = spkr_i[0].split("sv_")[-1].split(".")[0].split("_")[0]

        spkr_i_utter_path = spkr_i[0]
        utters = np.load(spkr_i_utter_path)

        # randomly picking an utterance for a speaker
        idx = random.randint(0, utters.shape[0]-1)

        # speaker utterance
        trim_len = hp.m_avc.s2.mul_32_utter_len
        spkr_i_utter = torch.tensor(utters[idx, :trim_len, :]).unsqueeze(0).to(hp.general.device)

        # speaker embeddings
        spkr_i_embs = spkr_i[1]
        spkr_i_embs = torch.tensor(spkr_i_embs).unsqueeze(0).to(hp.general.device)

        # again running the loop on metadata to produce AxB number of cross utterences
        for spkr_j in metadata:

            # if spkr_j == spkr_i:
            #     continue

            spkr_j_embs = torch.from_numpy(spkr_j[1][np.newaxis, :]).to(hp.general.device)
            spkr_j_name = spkr_j[0].split("/")[-1].split("_")[1]

            with torch.no_grad():
                _, x_identic_psnt, _ = auto_vc_model(spkr_i_utter, spkr_i_embs, spkr_j_embs)

            uttr_trg = x_identic_psnt[0, :, :].cpu().numpy()

            print("Processed ", '{}x{}'.format(spkr_i_name, spkr_j_name))
            voice_cloned_spects.append(('{}x{}'.format(spkr_i_name, spkr_j_name), uttr_trg))

    # saving cross mel-specs to a file
    p1 = os.path.join(hp.general.project_root, hp.m_avc.gen.cross_mel_specs_path)
    file_path = os.path.join(p1, hp.m_avc.gen.cross_mel_specs_file)
    with open(file_path, 'wb') as handle:
        pickle.dump(voice_cloned_spects, handle)

    print("Success, voice cloned spect saved!!")


# quick test, below code will not be executed when the file is imported
# it runs only when this file is directly executed
if __name__ == '__main__':
    from strings.constants import hp
    create_mel_specs_per_speaker(hp)
