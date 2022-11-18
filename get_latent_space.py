import torch
from tqdm.auto import tqdm
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


