import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from tqdm.auto import tqdm
import os

# dir="C:/Users/tobia/Desktop/Simula/ecg_gen/output"
# test_data=glob.glob(dir + '/*.csv')
# for Set in tqdm(test_data):
#     combined_df=pd.read_csv(Set,index_col=0)



def get_lists(input_dir):
    """
    Takes CSV files from input dir and returns MSELoss,pointwise loss and paired leads for ploting overlapping leads
    """
    test_data=glob.glob(input_dir + '/*.csv')
    all_loss_list=[]
    all_loss_over_time_list=[]
    all_lead_pair_list=[]
    for Set in tqdm(test_data):
        combined_df=pd.read_csv(Set,index_col=0)
        #list of averages losses betwen lead pairs
        loss_list=[]
        #list of losses between lead pairs over time
        loss_over_time_list=[]
        #lead pair list for plotting lead pairs on top of each other
        lead_pair_list=[]
        criterion = nn.MSELoss()
        #going over all columns in combined dataframe
        for i in range(12):
            t1=(torch.Tensor(combined_df.iloc[:,i]))
            t2=(torch.Tensor(combined_df.iloc[:,i+12]))
            lead_pair=(t1,t2)
            lead_pair_list.append(lead_pair)
            loss=criterion(t1,t2)
            loss_list.append(loss)
            loss_over_time=(t1-t2)**2
            loss_over_time_list.append(loss_over_time) 
        all_loss_list.append(loss_list)
        all_loss_over_time_list.append(loss_over_time_list)
        all_lead_pair_list.append(lead_pair_list)
    return all_loss_list,all_loss_over_time_list,all_lead_pair_list


#all_loss_list,all_loss_over_time_list,all_lead_pair_list=get_lists("C:/Users/tobia/Desktop/Simula/ecg_gen/output")

def overlapping_leads(title="_",version=None,lead_pairs=None,path=None):
  """
  used a list of paired leads to plot them overlapping
  """
  fig,axs=plt.subplots(12,sharex=True,sharey=True,figsize=(15,8))
  plt.ylabel("mV",x=0.5,y=7.5)
  plt.xlabel("timesteps")
  plt.title("Real and predicted lead pairs I,II,III,aVR,aVL,aVF,v1,v2,v3,v4,v5,v6", x=0.5,y=14.5)
  save_dir=str(Path(path).joinpath("loss_visual"))
  os.makedirs(save_dir,exist_ok=True)
  for k,(real,synth) in enumerate(lead_pairs):
    axs[k].plot(real.tolist(),label="real",linewidth=0.5,c="g",alpha=1)
    axs[k].plot(synth.tolist(),label="prediction",linewidth=0.5,c="r",alpha=0.5)
  plt.legend(loc=1, bbox_to_anchor=(1,15.7))
  plt.savefig(f"{save_dir}/{title}_overlap_{version}.pdf", format="pdf", bbox_inches="tight")
  plt.close()

    
def plot_ECG_loss_over_time(title="_",input=None,path=None,version=None,label=None):
  """
  Takes a list of pointwise losses and returns as plot. Adds the MSELoss to each subplot as label 
  """
  save_dir=str(Path(path).joinpath("loss_visual"))
  os.makedirs(save_dir,exist_ok=True)
  fig, axs = plt.subplots(12,sharex=True,sharey=True,figsize=(15,8))
  plt.ylabel("MSEloss",x=0.5,y=7.5)
  plt.xlabel("timesteps")
  plt.title("Loss between real and predicted lead pairs I,II,III,aVR,aVL,aVF,v1,v2,v3,v4,v5,v6",x=0.5,y=14.5)
  for k,item in enumerate(input):
    loss_label=str(round(label[k].item(),4))
    axs[k].plot(item.squeeze(),label="average_loss:"+loss_label)
    axs[k].legend(loc='upper right')
  #plt.legend(loc=1, bbox_to_anchor=(1,15.7))
  plt.savefig(f"{save_dir}/{title}_LossOverTime_{version}.pdf", format="pdf", bbox_inches="tight")
  plt.close()



#plot_ECG_loss_over_time(loss_over_time_list)

def ecgloss_visual(title,input_dir,output_dir):
  print("calculating...")
  x,y,z=get_lists(input_dir)
  print("Overlapping leads...")
  for k,sub in enumerate(z):
    overlapping_leads(title,version=k,lead_pairs=sub,path=output_dir)
  print("Plotting loss over time...")
  for k,i in enumerate(range(len(x))):
    plot_ECG_loss_over_time(title,input=y[i],path=output_dir,version=k,label=x[i])
  print("visualization done")


#input_dir="C:/Users/tobia/Desktop/Simula/ecg_gen/output/"
#ecgloss_visual(title="syn",input_dir=input_dir,output_dir=input_dir)

