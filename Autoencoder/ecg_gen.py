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

# parser = argparse.ArgumentParser()
#file_path=pathlib.PurePath(__file__).parent
#==============================
# Directory and file handling
#==============================
# parser.add_argument("--out_dir",default="C:/Users/tobia/Desktop/Simula/studio_output")
# parser.add_argument("--input_dir",default="C:/Users/tobia/Desktop/Simula/studio_input")
# parser.add_argument("--model_type",default="syn",choices=["syn","Normal","Patho"])
# parser.add_argument("--model_version",default="6",choices=["0","1","6","10"],help="model state 6 is usualy the best")
# parser.add_argument("--action",default="inference", type=str, help="Select an action to run", choices=["train", "retrain", "inference", "check"])
# parser.add_argument("--Epochs",default=3, type=int, help="Select Epochs to run for")
# parser.add_argument("--Batch_size",default=8, type=int, help="Select BAtch size, default is 32")
# parser.add_argument("--Lr",default=0.001, type=int, help="Select Learning rate")
# parser.add_argument("--API_key",default=None, type=str, help="Add your WandB API key to send data to WandB")
# parser.add_argument("--down_state",default=True, type=bool, help="Specify to download a fresh model state")

#7a8ee9d41cc2d51eb77fd795e14f74a215e63c2d

# opt=parser.parse_args()

#==============================
# Get model state
#==============================
def get_model_state(type,version,input_folder):
    path=f"https://github.com/t-willi/LargeFileStorage/blob/main/Simula_model_checkpoints/{type}_v{version}.pt?raw=true"
    print("downloading model state dict")
    get_state(state_path=path,save_path=input_folder)

#==============================
# Initialize model
#==============================
def init_model(type=None,version=None,input_dir=None):
    """
    Choose from type:Syn,Patho,Normal and version:best,last. 
    """
    model=Pulse2pulseGenerator()
    get_model_state(type,version,input_dir)
    model.load_state_dict(torch.load(input_dir+"/model_state/model.pt",map_location="cpu"))
    return model

#==============================
# Inference mode
#==============================

def run_inference(type,version,input_path,output_path=None,normalize=None):
    #############################################
    # set input output folders
    input_dir=input_path
    if not output_path:
        output_path=str(pathlib.PurePath(input_dir).parent.joinpath("output"))
        os.makedirs(output_path,exist_ok = True)
    output_dir=output_path
    #############################################
    model=init_model(type,version,input_dir)
    maximum_value=1
    if normalize:
        print("looking for max value")
        maximum_value=find_max(input_dir)
        print(f"All data will be normalized by{maximum_value}")
    dataset=CD(input_dir,split=False,max_value=maximum_value)
    print("data was generated")
    for k,(i) in enumerate(dataset):
        input,output=predictions(dataset=i,model=model,upscale=maximum_value,job="inference")
        print("predictions were made")
        ecg_df=ecg(input,output,title=k,path=output_dir)
        scaled_output=output/1000
        scaled_output.to_csv(output_dir+f"/ecg{k}.csv")

def Syn_trans(type="syn",input_path=None,output_path=None,normalize=None):
    """
    function takes in input_path leading to folder with ecg.csv files. Files should contain 8 columns and 5000 rows of data plus an index.
    The first columns will be used to generate 12 leads, the output will be 24 signals, left the input data and right the generated data.
    If no output folder is specified output folder will be created in same dir as input folder
    If there is a need to normalize the data to [1,-1] set normalize to True.
    For now user is required to run !pip install ecg_plot to install plotting package.
    """
    run_inference(type,"6",input_path,output_path,normalize=normalize)


#Syn_trans(type="syn",input_path="C:/Users/tobia/Desktop/Simula/ecg_gen/input",normalize=True)

#output_path="C:/Users/tobia/Desktop/Simula/ecg_gen/output"