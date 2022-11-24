from tqdm.auto import tqdm
import torch
from torch import nn
def train(model,train_loader,val_loader,test_dataset, criterion, optimizer, opt,predictions=None,ecg=None):
  device=opt.device
  output_dir=opt.out_dir
  # Make the loss and optimizer
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
  #wandb.watch(model, criterion, log="all")
  for epoch in tqdm((range(opt.epochs))):
    train_loss=0
    for batch,(X,y) in tqdm((enumerate(train_loader))):
      # Forward pass ➡
      X, y = X.to(device), y.to(device)
      #print(f"shape of input{x.shape},shape of label_y{y.shape}") 
      model.train()
      #print(f"x.shape is{X.max()} y.shape is {y.max()}")
      output=model(X)
      #print(f"output is {output.max()}")
      #print(f"shape of model_output_raw{output.shape}") 
      # output=torch.reshape(output,(config.batch_size, 1, 7, 5000))
      loss = criterion(output,y)
      train_loss += loss
      # Backward pass ⬅
      optimizer.zero_grad()
      loss.backward()
      # Step with optimizer
      optimizer.step()
    #average loss per batch
    train_loss /= len(train_loader)


    val_loss = 0
    model.eval()
    with torch.inference_mode():
      for batch,(X,y) in tqdm(enumerate(val_loader)):
        #print("doing test loop")
        X, y = X.to(device), y.to(device)
        val_pred = model(X)
        # val_pred=torch.reshape(val_pred,(config.batch_size, 1, 7, 5000))
        loss=criterion(val_pred,y)
        val_loss += loss
      val_loss /= len(val_loader)  
    #   wandb.log({"train_loss": train_loss, 
    #              "val_loss": val_loss,
    #              "Epoch":epoch})
      

    if (epoch) % 1==0:
      df_input,df_output=predictions(test_dataset,model)
      model.to(device)
      #plotting the ECG and creating the combined DF
      combined_df=ecg(df_input,df_output,path=output_dir)
      #saving combined DF as table on wandB
      #input_prediction_table = wandb.Table(dataframe=combined_df)
    #   wandb.log({"ECG": wandb.Image(str(ecg_dir_file))})
    #   wandb.log({"Input and predictions": input_prediction_table}) 
      torch.save(model.state_dict(),output_dir)
      scaled_output=df_output/1000
      scaled_output.to_csv(opt.out_dir+f"/ecg{epoch}.csv")

