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
parser.add_argument("--Epochs",default=3, type=int, help="Select Epochs to run for")
parser.add_argument("--Batch_size",default=8, type=int, help="Select BAtch size, default is 32")
parser.add_argument("--Lr",default=0.001, type=int, help="Select Learning rate")
parser.add_argument("--API_key",default=None, type=str, help="Add your WandB API key to send data to WandB")
#7a8ee9d41cc2d51eb77fd795e14f74a215e63c2d

opt=parser.parse_args()
#==============================
# Initialize model
#==============================
def init_model(action=opt.action,type=None,version=None):
    """
    Choose from type:Syn,Patho,Normal and version:best,last. 
    """
    state_path=file_path.parent.joinpath("checkpoints","Autoencoder",f"{opt.model_type}_{opt.model_version}")
    model=Pulse2pulseGenerator()
    if action == "inference" or action == "retrain":
        print(f"Checkpoint loaded since job is {action}")
        model.load_state_dict(torch.load(str(state_path),map_location="cpu"))
    return model

torch.save({
    "model_state_dict": init_model().state_dict(),
}, opt.out_dir+"/model_state.pt")
print("saved model")
#==============================
# Device handling
#==============================
#torch.cuda.set_device(opt.device_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#==============================
# Inference mode
#==============================
print("created output folder")
def run_inference(data_dir=opt.input_dir):
    dataset=CD(data_dir,split=False)
    print("data was generated")
    for k,(i) in enumerate(dataset):
        input,output=predictions(dataset=i,model=init_model(),upscale=5011,job=opt.action)
        print("predictions were made")
        ecg_df=ecg(input,output,title=k,path=opt.out_dir)
        scaled_output=output/1000
        scaled_output.to_csv(opt.out_dir+f"/ecg{k}.csv")

#==============================
# Train/retrain mode
#==============================
def run_train(opt=opt,device=device):
    if opt.API_key:
        import wandb
        print("appending API key")
        wandb.login(key=opt.API_key)
        wandb.init()
    data_dir=opt.input_dir
    model=init_model()
    train_loader=ML(CD(data_dir,split=True,target="train"),opt.Batch_size)
    val_loader=ML(CD(data_dir,split=True,target="val"),opt.Batch_size)
    test_dataset=CD(data_dir,split=True,target="test")
    train(model,train_loader,val_loader,test_dataset,opt=opt,device=device)

if __name__ == "__main__":
    # Train or retrain or inference
    if opt.action == "train":
        print("Training process is strted..!")
        run_train()
        pass
    elif opt.action == "retrain":
        print("Retrainning process is strted..!")
        run_train()
        pass
    elif opt.action == "inference":
        run_inference()
        print("Inference process is started..!")
        pass