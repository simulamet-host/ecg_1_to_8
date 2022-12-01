# ECG_1to12 lead conversion

In this repository you find code to translate a "LeadI" signal into a full 12 Lead ECG.
To do this run the Syn_trans function in Autoencoder/ecg_gen.py. 
Set the model_checkpoint_dtype, an input directory containing your LeadI.csv tables and choose the unit your data is in,
aswell as you need to normalize it to a range between (-1,1). 
Running Syn_trans will output the predicted 12 Lead ECG as a CSV table and a ecg_plot.
If you are testing out the accuracy of the model you can use the Visualization tools, provided in Visualization/ecg_loss.py.
Here after specifying the input(the output of Syn_trans) and running ecgloss_visual, you get plots, showing the average loss between predictedand real signal,
and the loss over time, showing where the model predicts well and where it does not. Link to this page is as followed_
[Github-flavored Markdown](https://github.com/simulamet-host/ecg_1_to_8/tree/side)
