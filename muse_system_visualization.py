import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

columns_to_analyze = ["R_PeakAmpl_II", "R_PeakAmpl_III", "R_PeakAmpl_V1", "R_PeakAmpl_V2",
                      "R_PeakAmpl_V3", "R_PeakAmpl_V4", "R_PeakAmpl_V5", "R_PeakAmpl_V6",
                      "T_PeakAmpl_II", "T_PeakAmpl_III", "T_PeakAmpl_V1", "T_PeakAmpl_V2",
                      "T_PeakAmpl_V3", "T_PeakAmpl_V4", "T_PeakAmpl_V5", "T_PeakAmpl_V6",
                      "STJ_II", "STJ_III", "STJ_V1", "STJ_V2", "STJ_V3", "STJ_V4", "STJ_V5", "STJ_V6",
                      "STM_II", "STM_III", "STM_V1", "STM_V2", "STM_V3", "STM_V4", "STM_V5", "STM_V6",
                      "STE_II", "STE_III", "STE_V1", "STE_V2", "STE_V3", "STE_V4", "STE_V5", "STE_V6",
                      "STE_II", "STE_III", "STE_V1", "STE_V2", "STE_V3", "STE_V4", "STE_V5", "STE_V6",
                      "QT_Interval", "QRSDuration", "AvgRRInterval", "PR_Interval"]
gan_mixed_patient_prefix = "M2m_"
gan_ptb_patient_prefix = "M2p_"
gan_synthetic_patient_prefix = "M2s_"
synthetic_patient_prefix = "s_"
ptb_patient_prefix = "p_"

def interpret_file(filename: str, target_folder: str = "plots"):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    col1_name = f"generated"
    col2_name = f"real"
    def create_figure(title: str, data: pd.DataFrame):
        scatterplot = sns.scatterplot(x=col1_name, y=col2_name, data=data, s=3)
        scatterplot.set(title=title)
        fig = scatterplot.get_figure()
        fig.savefig(os.path.join(target_folder, f"{title}.png"))
        plt.close(fig)
    
    data = pd.read_csv(filename)
    gan_mixed_data = data[data["PatientID"].str.startswith(gan_mixed_patient_prefix)].sort_values(by="PatientID")
    gan_ptb_data = data[data["PatientID"].str.startswith(gan_ptb_patient_prefix)].sort_values(by="PatientID")
    gan_synthetic_data = data[data["PatientID"].str.startswith(gan_synthetic_patient_prefix)].sort_values(by="PatientID")
    ptb_data = data[data["PatientID"].str.startswith(ptb_patient_prefix)].sort_values(by="PatientID")
    synthetic_data = data[data["PatientID"].str.startswith(synthetic_patient_prefix)].sort_values(by="PatientID")

    for column in columns_to_analyze:
        concatenated_data_ptb = pd.DataFrame(data={col1_name: gan_ptb_data[column].values, col2_name: ptb_data[column].values})
        concatenated_data_mixed = pd.DataFrame(data={col1_name: gan_mixed_data[column].values, col2_name: ptb_data[column].values})
        concatenated_data_synthetic = pd.DataFrame(data={col1_name: gan_synthetic_data[column].values, col2_name: synthetic_data[column].values})
        create_figure(f"{column} PTB", concatenated_data_ptb)
        create_figure(f"{column} Mixed", concatenated_data_mixed)
        create_figure(f"{column} Synthetic", concatenated_data_synthetic)
        try:
            print(f"{column} Mean: {np.mean(gan_ptb_data[column].values)}")
            print(f"{column} Std: {np.std(gan_ptb_data[column].values)}")
            print(f"{column} PTB: {pearsonr(gan_ptb_data[column].values, ptb_data[column].values)}")
        except:
            pass
        try:
            print(f"{column} Mean: {np.mean(gan_mixed_data[column].values)}")
            print(f"{column} Std: {np.std(gan_mixed_data[column].values)}")
            print(f"{column} Mixed: {pearsonr(gan_mixed_data[column].values, ptb_data[column].values)}")
        except:
            pass
        try:
            print(f"{column} Mean: {np.mean(gan_synthetic_data[column].values)}")
            print(f"{column} Std: {np.std(gan_synthetic_data[column].values)}")
            print(f"{column} Synthetic: {pearsonr(gan_synthetic_data[column].values, synthetic_data[column].values)}")
        except:
            pass


interpret_file("12SL-complete.csv")
