"""
Step4 of Auto Voice Cloning is to create embeddings for each speaker and save as np files
"""

from s3_auto_voice_cloner.s5_auto_vc_train import TrainAutoVCNetwork
from strings.constants import hp

hp.m_avc.tpm.lambda_cd = 1
hp.m_avc.tpm.num_iters = 900
hp.m_avc.tpm.log_step = 50
hp.m_avc.tpm.dot_print = 5
hp.m_avc.tpm.checkpoint_interval = 200
hp.m_avc.tpm.lr = 0.001
hp.m_avc.tpm.reduce_lr_interval = 250
hp.m_avc.tpm.data_batch_size = 1

solver = TrainAutoVCNetwork(hp)

# start the training
auto_vc_model, lst_loss_tuple = solver.start_training()

print(5)
