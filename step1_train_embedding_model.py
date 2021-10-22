from strings.constants import hp

from s1_data_prep.audio_to_spectrogram import CreateSpectrogram
from s2_generalized_end2end_loss_GE2E.s4_train_embed_model import TrainEmbedModel

# step 1 prepare the spectrogram from the raw audio file
cr_obj = CreateSpectrogram(hp)
cr_obj.save_spectrogram_tisv()


# step2 of the Auto voice cloning is to: train the embedding model
# to get a user's embedding or 256 dim vector representing user's speaking style

hp.m_ge2e.training_epochs = 100
hp.m_ge2e.checkpoint_interval = 30
hp.m_ge2e.min_test_loss = 4


# creating training object
train_emb_model_obj = TrainEmbedModel(hp)

# training the model
model, train_loss, test_loss = train_emb_model_obj.train_model(lr_reduce=30, epoch_print=10, dot_print=1)
print(2)

