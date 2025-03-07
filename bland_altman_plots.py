import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib.font_manager import FontProperties


muse_df = pd.read_csv("12SL-ecg.csv")

muse_df.set_index('PatientID', inplace=True)


parameters = ['TFull_Area_V1', \
              'S_PeakAmpl_V1', 'S_PeakAmpl_V2', 'S_PeakAmpl_V3', 'S_PeakAmpl_V5', 'S_PeakAmpl_V6', \
              'STM_V1', 'STM_V2', 'STM_V3', 'STM_V5', 'STM_V6', \
              'T_PeakAmpl_V1', 'T_PeakAmpl_V2', 'T_PeakAmpl_V3', 'T_PeakAmpl_V5', 'T_PeakAmpl_V6',\
              'R_PeakAmpl_V1', 'R_PeakAmpl_V2', 'R_PeakAmpl_V3', 'R_PeakAmpl_V5', 'R_PeakAmpl_V6']

param_mapping = {
    'TFull_Area_V1': 'Total Full Area in lead V1',
    'S_PeakAmpl_V1': 'S Amplitude in lead V1',
    'S_PeakAmpl_V2': 'S Amplitude in lead V2',
    'S_PeakAmpl_V3': 'S Amplitude in lead V3',
    'S_PeakAmpl_V5': 'S Amplitude in lead V5',
    'S_PeakAmpl_V6': 'S Amplitude in lead V6',
    'STM_V1': 'ST Morphology in lead V1',
    'STM_V2': 'ST Morphology in lead V2',
    'STM_V3': 'ST Morphology in lead V3',
    'STM_V5': 'ST Morphology in lead V5',
    'STM_V6': 'ST Morphology in lead V6',
    'T_PeakAmpl_V1': 'T Amplitude in lead V1',
    'T_PeakAmpl_V2': 'T Amplitude in lead V2',
    'T_PeakAmpl_V3': 'T Amplitude in lead V3',
    'T_PeakAmpl_V5': 'T Amplitude in lead V5',
    'T_PeakAmpl_V6': 'T Amplitude in lead V6',
    'R_PeakAmpl_V1': 'R Amplitude in lead V1',
    'R_PeakAmpl_V2': 'R Amplitude in lead V2',
    'R_PeakAmpl_V3': 'R Amplitude in lead V3',
    'R_PeakAmpl_V5': 'R Amplitude in lead V5',
    'R_PeakAmpl_V6': 'R Amplitude in lead V6',
}

network = 'gan'  # gan, unet


for parameter in parameters:
    mapped_param = param_mapping.get(parameter, parameter)  

    for lead in [1, 2]:  
        list_real = []
        list_generated = []

        for ecg in range(8562, 9514):
            real = muse_df.loc[f'dataset_{ecg}', parameter]

            if lead == 1:
                generated = muse_df.loc[f'gan_{lead}lead_{ecg}', parameter]
            else:
                generated = muse_df.loc[f'gan_{lead}leads_{ecg}', parameter]

            list_real.append(real)
            list_generated.append(generated)

        list_real = np.array(list_real)
        list_generated = np.array(list_generated)

        fig, ax = plt.subplots()

        ax.grid(False)
        font_title = FontProperties()
        font_title.set_family('sans-serif')
        font_title.set_size(18)

        font_axis_labels = FontProperties()
        font_axis_labels.set_family('sans-serif')
        font_axis_labels.set_size(12)

        font_axis_labels1 = FontProperties()
        font_axis_labels1.set_family('sans-serif')
        font_axis_labels1.set_size(14)

        diff = list_generated - list_real

        r_generated, _ = pearsonr(list_generated, list_real)
        r_diff, _ = pearsonr(diff, list_real)

        sns.scatterplot(x=list_real, y=diff, color="steelblue", s=20)

        mean_diff = diff.mean()
        std_diff = diff.std()

        plt.axhline(mean_diff, color='gray', linestyle='--')
        plt.axhline(mean_diff + 1.96 * std_diff, color='gray', linestyle='--')
        plt.axhline(mean_diff - 1.96 * std_diff, color='gray', linestyle='--')

        plt.text(plt.xlim()[1] + 0.05, mean_diff, 'mean', fontsize=10, color='gray')
        plt.text(plt.xlim()[1] + 0.05, mean_diff + 1.96 * std_diff, '+1.96 SD', fontsize=10, color='gray')
        plt.text(plt.xlim()[1] + 0.05, mean_diff - 1.96 * std_diff, '-1.96 SD', fontsize=10, color='gray')

        plt.annotate(f'R² = {r_diff**2:.2f}', xy=(plt.xlim()[0], plt.ylim()[0]), xytext=(10, 10),
                     textcoords='offset points', ha='left', va='bottom', fontsize=14)

        plt.subplots_adjust(left=0.15, right=0.85)

        plt.xlabel('Real data (µV)', fontproperties=font_axis_labels1)
        plt.ylabel('Reconstruction error (generated - real)', fontproperties=font_axis_labels1)
        plt.xticks(fontproperties=font_axis_labels)
        plt.title(f'{mapped_param}', fontproperties=font_title)

        plt.savefig(f'bland_altman/{parameter}_{network}_lead{lead}.pdf')
        plt.close(fig)
