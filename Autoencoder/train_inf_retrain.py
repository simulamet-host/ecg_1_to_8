import argparse
import numpy as np
import torch
from torch import nn
import glob
import pathlib
import os
from pathlib import Path
from Autoencoder.models.pTOP import *
#from models.pTOP import *
from Autoencoder.utils.get_states import get_state
from Autoencoder.utils.dataloader_csv import Custom_dataset_CSV as CD
from Autoencoder.utils.dataloader_csv import make_loader as ML

from Autoencoder.utils.get_predictions import get_pred_12lead as predictions
from Autoencoder.utils.get_ecgs import plotECG_12Lead as ecg
from Autoencoder.utils.train import train

seed=42
np.random.seed(seed)  # numpy random generator
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
file_path=pathlib.PurePath(__file__).parent
#==============================
# Directory and file handling
#==============================
parser.add_argument("--out_dir",default="C:/Users/tobia/Desktop/Simula/studio_output")
parser.add_argument("--input_dir",default="C:/Users/tobia/Desktop/Simula/studio_input")
parser.add_argument("--model_type",default="syn",choices=["syn","Normal","Patho"])
parser.add_argument("--model_version",default="6",choices=["0","1","6","10"],help="model state 6 is usualy the best")
parser.add_argument("--action",default="inference", type=str, help="Select an action to run", choices=["train", "retrain", "inference", "check"])
parser.add_argument("--Epochs",default=3, type=int, help="Select Epochs to run for")
parser.add_argument("--Batch_size",default=8, type=int, help="Select BAtch size, default is 32")
parser.add_argument("--Lr",default=0.001, type=int, help="Select Learning rate")
parser.add_argument("--API_key",default=None, type=str, help="Add your WandB API key to send data to WandB")
parser.add_argument("--down_state",default=True, type=bool, help="Specify to download a fresh model state")
parser.add_argument("--unit",default="µV", type=str, help="Specify the unit the ecg is in")
parser.add_argument("--max_value",default=5011, type=str, help="Specify the unit the ecg is in")



#7a8ee9d41cc2d51eb77fd795e14f74a215e63c2d

opt=parser.parse_args()

#==============================
# Get model state
#==============================
def get_model_state(opt):
    path=f"https://github.com/t-willi/LargeFileStorage/blob/main/Simula_model_checkpoints/{opt.model_type}_v{opt.model_version}.pt?raw=true"
    print("downloading model state dict")
    get_state(state_path=path,save_path=opt.input_dir)

#==============================
# Initialize model
#==============================
def init_model(opt,type=None,version=None):
    """
    Choose from type:Syn,Patho,Normal and version:best,last. 
    """
    #state_path=file_path.parent.joinpath("checkpoints","Autoencoder",f"{opt.model_type}_{opt.model_version}")
    input_dir=opt.input_dir
    action=opt.action
    load_model=opt.down_state
    model=Pulse2pulseGenerator()
    if action == "inference" or action == "retrain":
        if load_model:
            get_model_state(opt)
        print(f"Checkpoint loaded since job is {action}")
        model.load_state_dict(torch.load(input_dir+"/model_state/model.pt",map_location="cpu"))
    return model

#==============================
# Device handling
#==============================
#torch.cuda.set_device(opt.device_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#==============================
# Inference mode
#==============================
print("created output folder")
def run_inference(opt):
    data_dir=opt.input_dir
    dataset=CD(data_dir,split=False)
    model=init_model(opt)
    print("data was generated")
    for k,(i) in enumerate(dataset):
        input,output=predictions(dataset=i,model=model,upscale=5011,job=opt.action)
        print("predictions were made")
        ecg_df=ecg(input,output,title=f"_{opt.action}_{k}",path=opt.out_dir,unit=opt.unit)
        scaled_output=output
        if opt.unit == "µV":
            scaled_output=output/1000
        scaled_output.to_csv(opt.out_dir+f"/ecg_{opt.action}_{k}.csv")

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
    model=init_model(opt)
    train_loader=ML(CD(data_dir,max_value=opt.max_value,split=True,target="train"),opt.Batch_size)
    val_loader=ML(CD(data_dir,max_value=opt.max_value,split=True,target="val"),opt.Batch_size)
    test_dataset=CD(data_dir,max_value=opt.max_value,split=True,target="test")
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
        run_inference(opt)
        print("Inference process is started..!")
        pass