import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
def get_latent_space(dataloader=None,FE=None,model=None):
  """
  takes dataloader a trained feature extractor(7to7 leads) and a trained model(1to7 leads) and returns latent spaces for both
  """
  dataloader=dataloader
  FE=FE
  model=model
  Latent_spaces_real = torch.empty((1,100), dtype=torch.float32)
  Latent_spaces_predicted = torch.empty((1,100), dtype=torch.float32)
  for x,y in tqdm(dataloader):
    model.eval()
    FE.eval()
    with torch.inference_mode():
      #get latent space of real data
      output_real=FE(y,LS=True)
      Latent_spaces_real = torch.cat((Latent_spaces_real, output_real.detach().cpu()), 0)
      #get latent space of predicted data
      output_predicted=model(x)
      output_predicted=FE(output_predicted,LS=True)
      Latent_spaces_predicted = torch.cat((Latent_spaces_predicted, output_predicted.detach().cpu()), 0)
  return Latent_spaces_real,Latent_spaces_predicted


def TSNE_plot(dataset1,dataset2):
  random_state=42
  result1 = TSNE(n_components=2, learning_rate='auto',init='random',random_state=random_state, perplexity=100).fit_transform(Latent_spaces_real)
  result2 = TSNE(n_components=2, learning_rate='auto',init='random',random_state=random_state, perplexity=100).fit_transform(Latent_spaces_real)
  df1,df2=pd.DataFrame(result1),pd.DataFrame(result2)
  df1.columns=["x","y"]
  df2.columns=["x","y"]
  plt.figure(figsize=(8,8))
  fig, ax = plt.subplots()
  ax.scatter(y=df1["x"],x=df1["y"],c="g",alpha=1,label="Real Data")
  ax.scatter(y=df2["x"],x=df2["y"],c="b",alpha=0.2,label="Predicted Data")
  ax.legend()

def TSNE_plots(dataset_list):
  random_state=42
  plt.figure(figsize=(12,12))
  fig, ax = plt.subplots()
  for k,entry in enumerate(dataset_list):
    print(f"running TSNE for entry {k}")
    result = TSNE(n_components=2, learning_rate='auto',init='random',random_state=random_state, perplexity=100).fit_transform(entry)
    df=pd.DataFrame(result,columns=[["x","y"]])
    ax.scatter(y=df["x"],x=df["y"],alpha=0.2,label=f"set{k}")
    ax.legend()
