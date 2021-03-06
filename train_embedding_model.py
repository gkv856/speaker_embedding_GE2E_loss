from strings.constants import hp

from embedding_model_GE2E.s0_audio_to_spectrogram import CreateSpectrogram
from embedding_model_GE2E.s4_train_embed_model import TrainEmbedModel

# step 1 prepare the spectrogram from the raw audio file
# hp.raw_audio.raw_audio_path = "static/raw_data/librispeech_test-other"
cr_obj = CreateSpectrogram(hp, verbose=True)
cr_obj.save_spectrogram_tisv()


# step 1 prepare the spectrogram from the raw audio file
hp.raw_audio.raw_audio_path = "static/raw_data/librispeech_test-other"
cr_obj = CreateSpectrogram(hp, verbose=True)
cr_obj.save_spectrogram_tisv()

# step2 of the Auto voice cloning is to: train the embedding model
# to get a user's embedding or 256 dim vector representing user's speaking style

# change these hyper parameters as per your training needs
hp.m_ge2e.training_epochs = 100
hp.m_ge2e.checkpoint_interval = 3
hp.m_ge2e.min_test_loss = 4


hp.m_ge2e.training_N = 2
hp.m_ge2e.training_M = 16

hp.m_ge2e.test_N = 2
hp.m_ge2e.test_M = 16

# creating training object
train_emb_model_obj = TrainEmbedModel(hp)

# training the model
model, train_loss, test_loss = train_emb_model_obj.train_model(lr_reduce=3, epoch_print=2, dot_print=1)
print(2)
