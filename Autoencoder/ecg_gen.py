import numpy as np
import torch
from torch import nn
import glob
import pathlib
import os
from pathlib import Path
from Autoencoder.models.pTOP import *
from Autoencoder.utils.get_states import get_state
from Autoencoder.utils.dataloader_csv import Custom_dataset_CSV as CD
from Autoencoder.utils.find_max import find_max 
from Autoencoder.utils.get_predictions import get_pred_12lead as predictions
from Autoencoder.utils.get_ecgs import plotECG_12Lead as ecg
from Autoencoder.utils.train import train

seed=42
np.random.seed(seed)  # numpy random generator
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#==============================
# Get model state
#==============================
def get_model_state(kind,version,input_folder):
    path=f"https://github.com/t-willi/LargeFileStorage/blob/main/Simula_model_checkpoints/{kind}_v{version}.pt?raw=true"
    print(f"Downloading model state dict: {kind}_{version}")
    get_state(state_path=path,save_path=input_folder)

#==============================
# Initialize model
#==============================
def init_model(kind=None,version=None,input_dir=None):
    model=Pulse2pulseGenerator()
    get_model_state(kind,version,input_dir)
    model.load_state_dict(torch.load(input_dir+"/model_state/model.pt",map_location="cpu"))
    return model

#==============================
# Inference mode
#==============================

def run_inference(kind,version,input_path,output_path,normalize,unit):
    #############################################
    # set input output folders  
    input_dir=input_path
    if not output_path:
        output_path=str(pathlib.PurePath(input_dir).parent.joinpath("output"))
        os.makedirs(output_path,exist_ok = True)
    output_dir=output_path
    #############################################
    model=init_model(kind,version,input_dir)
    maximum_value=1
    if normalize:
        print("Looking for global max value in data")
        maximum_value=int(find_max(input_dir))
        print(f"Data will be normalized by: {maximum_value}")
    print(f"Data normalized by {maximum_value}")
    dataset=CD(input_dir,split=False,max_value=maximum_value)
    print(f"Dataset complete ,unit specified: {unit}")
    for k,(i) in enumerate(dataset):
        input,output=predictions(dataset=i,model=model,upscale=maximum_value,job="inference")
        print("predictions were made")
        unit=unit
        ecg_df=ecg(input,output,title=k,path=output_dir,unit=unit)
        ecg_df.to_csv(output_dir+f"/ecg{k}.csv")

def Syn_trans(kind="norm",version="best",input_path=None,output_path=None,normalize=True,unit="mV"):
    """
    Specify input folder with .csv files. Files should contain 8 columns and 5000 rows of data.
    Choose kind of model that fits best: "syn","norm","patho", the best version is already set.
    First column will be used to generate 12 leads, the output will be 24 leads, with the generated Leads located at the right.
    If data needs normalization to [1,-1] set normalize to True, otherwise data is normalized by 1.
    For now user is required to run !pip install ecg_plot to install plotting package.
    """
    run_inference(kind,version,input_path,output_path,normalize,unit)


Syn_trans(kind="syn",input_path="C:/Users/tobia/Desktop/Simula/ecg_gen/input/syn",output_path="C:/Users/tobia/Desktop/Simula/ecg_gen/output/syn",normalize=True,unit="µV")

#output_path="C:/Users/tobia/Desktop/Simula/ecg_gen/output"
