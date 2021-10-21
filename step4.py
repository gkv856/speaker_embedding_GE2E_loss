"""
Step4 of Auto Voice Cloning is to: create an embedding for each speaker and save as np files
"""
import torch

from s3_auto_voice_cloner.s1_create_emb_per_speaker import create_embbedings_per_speaker
from strings.constants import hp
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

utterances = create_embbedings_per_speaker(hp)

labels = ["s1", "s2", "s3", "s4"]

embs = []
for u in utterances:
    embs.append(u[1])

embeddings = torch.tensor(embs)

scatters = TSNE(n_components=2, random_state=0).fit_transform(embeddings.cpu().detach().numpy())
fig = plt.figure(figsize=(5, 5))

current_Label = labels[0]
current_Index = 0
for index, label in enumerate(labels[1:], 1):
    if label != current_Label:
        plt.scatter(scatters[current_Index:index, 0], scatters[current_Index:index, 1],
                    label='{}'.format(current_Label))
        current_Label = label
        current_Index = index

plt.scatter(scatters[current_Index:, 0], scatters[current_Index:, 1], label='{}'.format(current_Label))
plt.legend()
plt.tight_layout()
plt.show()

print(4)
