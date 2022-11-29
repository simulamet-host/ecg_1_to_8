
import pandas as pd
from pathlib import Path
import os
def plotECG_12Lead(df1=None,df2=None,title=None,path=None,createECG=True,unit="mV"):
  """
  takes two dataframes with identical columns, concats them and plots them as ecg using ecg_plot
  it also takes the first column of df1 and ads it to df1 if pad_df2 is True
  """
  import ecg_plot
  index=["real_I","real_II","real_III","real_aVR","real_aVL","real_aVF","real_v1","real_v2","real_v3","real_v4","real_v5","real_v6","real_I",
         "pred_II","pred_III","pred_aVR","pred_aVL","pred_aVF","pred_v1","pred_v2","pred_v3","pred_v4","pred_v5","pred_v6"]
  if createECG==True:
    ecg_path=path
    if Path(ecg_path).is_dir():
      None
    else:
        print(f"Did not find {ecg_path} directory, creating one...")
        #Path(ecg_path).mkdir(parents=True, exist_ok=False)
        os.mkdir(ecg_path)
  if unit == "µV":
    frames=[df1/1000,df2/1000]
  if unit == "mV":
    frames=[df1,df2]
  combined_df=pd.concat(frames,axis=1,join="outer",)
  if createECG is True:
    ecg_plot.plot(combined_df.values.T, sample_rate = 500,lead_index = index )
    ecg_plot.save_as_png(f'ecg{title}',ecg_path+"/")
  return combined_df

  