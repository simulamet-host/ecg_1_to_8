from tqdm.auto import tqdm
import pandas as pd
import glob
def find_max(data):
  """
  This takes a directory of pd.read_csv readable data and intterates trough all,
  finding the global max and min
  """
  from tqdm.auto import tqdm
  max_list=[]
  test_data=glob.glob(data + '/*.csv')
  #files=[*files_test,*files_train]
  for file in tqdm(test_data):
    temp_df=pd.read_csv(file,index_col=0)
    maximum=temp_df.max().max()
    max_list.append(maximum)
    maximum=max(max_list)
  return(maximum)

# maximum=find_max("C:/Users/tobia/Desktop/Simula/ecg_gen/input")
# print(maximum)