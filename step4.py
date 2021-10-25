"""
Step5 of Auto Voice Cloning is to: train the AutoVCNetwork
AutoVCmodel reads the spectrogram of a speaker and uses same user's embedding/speaking style to reconstruct the
same audio
https://arxiv.org/abs/1905.05879
"""

from s3_auto_voice_cloner.s5_auto_vc_train import TrainAutoVCNetwork
from strings.constants import hp

hp.m_avc.tpm.lambda_cd = 1
hp.m_avc.tpm.num_iters = 10
hp.m_avc.tpm.log_step = 2
hp.m_avc.tpm.dot_print = 1
hp.m_avc.tpm.checkpoint_interval = 2
hp.m_avc.tpm.lr = 0.001
hp.m_avc.tpm.reduce_lr_interval = 2
hp.m_avc.tpm.data_batch_size = 2

solver = TrainAutoVCNetwork(hp)

# start the training
auto_vc_model, lst_loss_tuple = solver.start_training()

print(5)
