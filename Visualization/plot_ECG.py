
import pandas as pd
from pathlib import Path
def plotECG_12Lead(df1=None,df2=None,title=None,pad_df2=True,path=None,createECG=True,scale=None):
  """
  takes two dataframes that contain each 12 ECG leads, concatenates them, returns a combined dataframe and
  a ecg plot using the ecg_plot package. ECG is saved with name(title) in directory(path)
  If createECG is false no ECG will be saved and if Scale is None the data will not be scaled. 
  The scale is used for data that is in µV instead of mV

  """
  import ecg_plot
  index=["real_I","real_II","real_III","real_aVR","real_aVL","real_aVF","real_v1","real_v2","real_v3","real_v4","real_v5","real_v6","real_I",
         "pred_II","pred_III","pred_aVR","pred_aVL","pred_aVF","pred_v1","pred_v2","pred_v3","pred_v4","pred_v5","pred_v6"]
  if createECG is True:
    ecg_path=path
    if Path(ecg_path).is_dir():
        print(f"{ecg_path} directory exists.")
    else:
        print(f"Did not find {ecg_path} directory, creating one...")
        Path(ecg_path).mkdir(parents=True, exist_ok=False)
  if scale:
    frames=[df1/1000,df2/1000]
  if scale is None:
    frames=[df1,df2]
  combined_df=pd.concat(frames,axis=1,join="outer",)
  if createECG is True:
    ecg_plot.plot(combined_df.values.T, sample_rate = 500,lead_index = index )
    ecg_plot.save_as_png(f'{title}',ecg_path)
  return combined_df

  