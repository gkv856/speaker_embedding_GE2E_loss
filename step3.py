"""
Step3 of Auto Voice Cloning is to test the embedding model
"""
from s2_generalized_end2end_loss_GE2E.s5_eval_model import calculate_ERR, plot_scatter
from strings.constants import hp
from s2_generalized_end2end_loss_GE2E.s2_model_GE2E_loss_speach_embed import  get_pre_trained_embedding_model

# loading a pre-trained model
model = get_pre_trained_embedding_model(hp)

# calculating ERR
calculate_ERR(model, hp, 4, 8)

# plotting speaker embeddings
plot_scatter(model, hp, 4, 16)

print(3)

print(4)
