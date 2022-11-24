import argparse
import numpy as np
import torch
from torch import nn
import glob
import pathlib
import os
from pathlib import Path
from models.pTOP import *
from utils.dataloader_csv import Custom_dataset_CSV as CD
from utils.dataloader_csv import make_loader as ML

from utils.get_predictions import get_pred_12lead as predictions
from utils.get_ecgs import plotECG_12Lead as ecg
from utils.train import train

torch.manual_seed(42)
np.random.seed(42)
parser = argparse.ArgumentParser()
file_path=pathlib.PurePath(__file__).parent
#==============================
# Directory and file handling
#==============================
parser.add_argument("--out_dir",default="C:/Users/tobia/Desktop/Simula/studio_output")
parser.add_argument("--input_dir",default="C:/Users/tobia/Desktop/Simula/studio_input")
parser.add_argument("--model_type",default="Syn",choices=["Syn","Normal","Patho"])
parser.add_argument("--model_version",default="best",choices=["best","last"])
parser.add_argument("--action",default="inference", type=str, help="Select an action to run", choices=["train", "retrain", "inference", "check"])
parser.add_argument("--Epochs",default=1, type=int, help="Select Epochs to run for")
parser.add_argument("--Batch_size",default=32, type=int, help="Select Epochs to run for")
parser.add_argument("--Lr",default=0.001, type=int, help="Select Learning rate")

opt=parser.parse_args()
#==============================
# Initialize model
#==============================
def init_model(action=opt.action,type=None,version=None):
    """
    Choose from type:Syn,Patho,Normal and version:best,last. 
    """
    state_path=file_path.joinpath("model_states",f"{opt.model_type}_{opt.model_version}")
    model=Pulse2pulseGenerator()
    if action == inference:
        model.load_state_dict(torch.load(str(state_path),map_location="cpu"))
    return model
#==============================
# Device handling
#==============================
torch.cuda.set_device(opt.device_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#==============================
# Inference mode
#==============================
print("created output folder")
def inference(data_dir=opt.input_dir):
    dataset=CD(data_dir,split=False)
    print("data was generated")
    for k,(i) in enumerate(dataset):
        input,output=predictions(dataset=i,model=init_model(),upscale=5011)
        print("predictions were made")
        ecg_df=ecg(input,output,title=k,path=opt.out_dir,scale=True)
        scaled_output=output/1000
        scaled_output.to_csv(opt.out_dir+f"/ecg{k}.csv")

inference()

#==============================
# Train mode
#==============================
def train():
    model=init_model()
    train_loader=ML(CD(data_dir=opt.input_dir,split=True,target="train"),opt.Batch_size)
    val_loader=ML(CD(data_dir=opt.input_dir,split=True,target="val"),opt.Batch_size)
    test_dataset=CD(data_dir=opt.input_dir,split=True,target="test")
    train(model,train_loader,val_loader,test_dataset,opt,predictions=None,ecg=None)



# if __name__ == "__main__":
#     # Train or retrain or inference
#     if opt.action == "train":
#         print("Training process is strted..!")
#         pass
#     elif opt.action == "retrain":
#         print("Retrainning process is strted..!")
#         pass
#     elif opt.action == "inference":
#         inference()
#         print("Inference process is started..!")
#         pass