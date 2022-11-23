import torch
from torch import nn
from models.pTOP import *
model=Pulse2pulseGenerator()
model.load_state_dict(torch.load("Autoencoder\model_states\synthetic_v6",map_location="cpu"))
print(model)
