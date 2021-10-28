## This is a WIP repo, final version will be released soon

### This repo is an End-to-End pytorch implementation of 
- 'AUTOVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss' paper
- 'Generalized End-to-End Loss for Speaker Verification' (GE2E loss) paper
- **Note**: Data used here is very elementary, please prepare you own data to train the model


### Modules included in this repo
- Voice Embedding Model: model, data, training, and pre-trained model
- GE2E implementation  
- AutoVC Zero-Shot voice style transfer


### Dependencies
- Python 3
- Numpy
- PyTorch
- librosa
- tqdm
- wavenet_vocoder ```pip install wavenet_vocoder```
  for more information, please refer to [this](https://github.com/r9y9/wavenet_vocoder)



### Pre-trained models (soon to be uploaded)

| Embedding Model (GE2E) | AutoVC | WaveNet Vocoder |
|----------------|----------------|----------------|
| [link](#)| [link](#) | [link](#) |


### Technically there are 7 steps to use this repo
**Steps will be detailed here very soon**

**Step1** Create spectrogram and train the Speaker Embedding model. Use the code from ```step1_train_embedding_model.py```

**Results Step 1 (Embedding model using GE2E loss)**
- With 4 Speakers 
- With 6 Speakers
- With 10 Speakers

*With this we are confident that our embedding model is working as expected*

![Speaker classification for 4 speakers](static/outputs/embedding_model/emb_04_spkr.png)
![Speaker classification for 4 speakers](static/outputs/embedding_model/emb_06_spkr.png)
![Speaker classification for 4 speakers](static/outputs/embedding_model/emb_10_spkr.png)


**Step2** Is about training the Auto Voice Clone (AVC Model)

**Step3**

**Step4**

**Step5**

**Step6**

**Step7**


### Papers used in this repo
- ```Generalized End-to-End Loss for Speaker Verification (https://arxiv.org/abs/1710.10467)```
- ```AUTOVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss (https://arxiv.org/abs/1905.05879)```

### Inspired from following github repo (BIG THANKS)
- ```https://github.com/auspicious3000/autovc```
- ```https://github.com/resemble-ai/Resemblyzer```

