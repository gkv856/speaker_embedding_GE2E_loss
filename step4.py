"""
Step5 of Auto Voice Cloning is to: train the AutoVCNetwork
AutoVCmodel reads the spectrogram of a speaker and uses same user's embedding/speaking style to reconstruct the
same audio
https://arxiv.org/abs/1905.05879
"""

from s3_auto_voice_cloner.s5_auto_vc_train import TrainAutoVCNetwork
from strings.constants import hp

hp.m_avc.tpm.lambda_cd = 1
hp.m_avc.tpm.num_iters = 20
hp.m_avc.tpm.log_step = 1
hp.m_avc.tpm.dot_print = 1
hp.m_avc.tpm.checkpoint_interval = 2
hp.m_avc.tpm.lr = 0.001
hp.m_avc.tpm.reduce_lr_interval = 5
hp.m_avc.tpm.data_batch_size = 2
hp.m_avc.tpm.norm_batch = True
hp.m_avc.tpm.use_dr = True
hp.m_avc.tpm.dr = 0.5
hp.m_avc.tpm.save_imgs = True

# resume training
hp.m_avc.gen.load_pre_trained_model = False
hp.m_avc.gen.best_model_path = "static/model_chk_pts/autovc/final_200.pth"
hp.m_avc.gen.st_epoch_cnt = 200

solver = TrainAutoVCNetwork(hp, absolute_path=False)

# start the training
auto_vc_model, lst_loss_tuple = solver.start_training(batched=False)

print(5)
