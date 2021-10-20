"""
Step2 of Auto Voice Cloning is to pre-train an embedding model
using Generalized End2End embedding GE2E
"""

from s2_generalized_end2end_loss_GE2E.s4_train_embed_model import TrainEmbedModel
from strings.constants import hp


hp.m_ge2e.training_epochs = 100
hp.m_ge2e.checkpoint_interval = 10
hp.m_ge2e.min_test_loss = 4

# creating training object
train_emb_model_obj = TrainEmbedModel(hp)

# training the model
model, train_loss, test_loss = train_emb_model_obj.train_model(lr_reduce=20, epoch_print=10, dot_print=1)
print(2)