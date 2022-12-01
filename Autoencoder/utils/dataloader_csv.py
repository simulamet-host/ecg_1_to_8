import glob
import pandas as pd
import torch
import numpy as np
class Custom_dataset_CSV():
    def __init__(self, data_dir,max_value=5011,column=3,split=True,target="train",size=1):
      #get all files from directory loaded in all_files list
      self.column=column
      self.max_value=max_value
      self.size=size
      #should shuffle the data here?
      self.files = glob.glob(data_dir + '/*.csv')
      self.len=int((len(self.files))*self.size)
      #print(f"len:{self.len}")
      self.cut1=int(self.len*0.8)
      #print(f"cut1:{self.cut1}")
      self.cut2=int(self.len*0.9)
      #print(f"cut2:{self.cut2}")
      self.train_files=self.files[0:self.cut1]
      self.test_files=self.files[self.cut1:self.cut2]
      self.val_files=self.files[self.cut2:self.len]
      self.target=target
      self.split=split

    def __len__(self):
      if self.split is True:
        if self.target == "train":
          return len(self.train_files)
        if self.target == "test":
          return len(self.test_files)
        if self.target == "val":
          return len(self.val_files)
      if self.split is not True:
        return len(self.files)

    def __getitem__(self,idx):
      header_8=["I", "II", "v1", "v2", "v3", "v4", "v5", "v6"]
      header_12=["I", "II","III","aVR","aVL","aVF", "v1", "v2", "v3", "v4", "v5", "v6"]
      #turn list of dataframes into Tensor
      if self.split is True:
        if self.target == "train":
          temp_df=pd.read_csv(self.train_files[idx],index_col=0,header=0)
        if self.target == "test":
          temp_df=pd.read_csv(self.test_files[idx],index_col=0,header=0)
        if self.target == "val":
          temp_df=pd.read_csv(self.val_files[idx],index_col=0,header=0)
      if self.split is not True:
        temp_df=pd.read_csv(self.files[idx],index_col=0,header=0)
      temp_df.index=[np.arange(0,5000)]
      if len(temp_df.columns) == 1:
        print("found only one column")
        print("adding columns for plotting purposes")
        for i in range(7):
          temp_df[i]=0
      if len(temp_df.columns) == 8:
        temp_df.columns=header_8
      else:
        temp_df.columns=header_12
      temp_df/=self.max_value
      #load input tensor
      temp_list_in=temp_df.loc[:,"I"]
      #temp_list_in=normalize([temp_list_in], norm="max")
      temp_tensor_in = torch.tensor(temp_list_in,dtype=torch.float32)
      temp_tensor_in=temp_tensor_in.unsqueeze(0)
      #load label Tensor
      temp_list_out=temp_df.loc[:,["II","v1","v2","v3","v4","v5","v6"]].values
      #temp_list_out=normalize([temp_list_out], norm="max")
      temp_tensor_out=torch.tensor(temp_list_out,dtype=torch.float32)
      temp_tensor_out=temp_tensor_out.T
      #combine input and label and output
      temp_tensor_pair= temp_tensor_in,temp_tensor_out
      return temp_tensor_pair

def make_loader(dataset,batch_size):
  from torch.utils.data import DataLoader
  loader = DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      drop_last=True
                      )
  return loader