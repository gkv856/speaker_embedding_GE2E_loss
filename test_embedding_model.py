from embedding_model_GE2E.s5_eval_model import calculate_ERR, plot_scatter
from strings.constants import hp
from embedding_model_GE2E.s2_model_GE2E_loss_speach_embed import  get_pre_trained_embedding_model

# loading a pre-trained model

# You can change the path of embedding model here

# hp.m_ge2e.best_model_path = "static/model_chk_pts/ge2e/final_epoch_1000_L_0.0390.pth"
model = get_pre_trained_embedding_model(hp, use_path_as_absolute=False)

# calculating ERR
calculate_ERR(model, hp, 4, 8)

plot_scatter(model, hp, 4, 32)

print(3)
