import torch
from torch import nn
import glob
import pathlib
from pathlib import Path
from models.pTOP import *
from utils.dataloader_csv import Custom_dataset_CSV as CD
from utils.get_predictions import get_pred_12lead as predictions
from utils.get_ecgs import plotECG_12Lead as ecg

file_path=pathlib.PurePath(__file__).parent

#initialize model
def init_model(version=6):
    state_path=file_path.joinpath("model_states",f"synthetic_v{version}")
    model=Pulse2pulseGenerator()
    model.load_state_dict(torch.load(str(state_path),map_location="cpu"))
    return model

#import data
data_dir=file_path.joinpath("input_data","Syn")

#get predictions
output_path=file_path.joinpath("ecg_outputs")
def generate_ecg(data_dir=str(data_dir)):
    dataset=CD(data_dir,split=False)
    print("data was generated")
    for k,(i) in enumerate(dataset):
        input,output=predictions(dataset=i,model=init_model(),upscale=5011)
        print("predictions were made")
        ecg_df=ecg(input,output,title=k,path=str(output_path),scale=True)
generate_ecg()



    







