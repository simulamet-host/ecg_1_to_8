import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib.font_manager import FontProperties

muse_df = pd.read_csv("12SL-ecg.csv")
muse_df.set_index('PatientID', inplace=True)

parameters = [
    'S_PeakAmpl_V1', 'S_PeakAmpl_V2', 'S_PeakAmpl_V3', 'S_PeakAmpl_V5', 'S_PeakAmpl_V6',
    'STM_V1', 'STM_V2', 'STM_V3', 'STM_V5', 'STM_V6',
    'T_PeakAmpl_V1', 'T_PeakAmpl_V2', 'T_PeakAmpl_V3', 'T_PeakAmpl_V5', 'T_PeakAmpl_V6',
    'R_PeakAmpl_V1', 'R_PeakAmpl_V2', 'R_PeakAmpl_V3', 'R_PeakAmpl_V5', 'R_PeakAmpl_V6'
]

param_mapping = {
    'S_PeakAmpl_V1': 'S Amplitude in Lead V1', 'S_PeakAmpl_V2': 'S Amplitude in Lead V2',
    'S_PeakAmpl_V3': 'S Amplitude in Lead V3', 'S_PeakAmpl_V5': 'S Amplitude in Lead V5',
    'S_PeakAmpl_V6': 'S Amplitude in Lead V6',
    'STM_V1': 'ST Morphology in Lead V1', 'STM_V2': 'ST Morphology in Lead V2',
    'STM_V3': 'ST Morphology in Lead V3', 'STM_V5': 'ST Morphology in Lead V5',
    'STM_V6': 'ST Morphology in Lead V6',
    'T_PeakAmpl_V1': 'T Amplitude in Lead V1', 'T_PeakAmpl_V2': 'T Amplitude in Lead V2',
    'T_PeakAmpl_V3': 'T Amplitude in Lead V3', 'T_PeakAmpl_V5': 'T Amplitude in Lead V5',
    'T_PeakAmpl_V6': 'T Amplitude in Lead V6',
    'R_PeakAmpl_V1': 'R Amplitude in Lead V1', 'R_PeakAmpl_V2': 'R Amplitude in Lead V2',
    'R_PeakAmpl_V3': 'R Amplitude in Lead V3', 'R_PeakAmpl_V5': 'R Amplitude in Lead V5',
    'R_PeakAmpl_V6': 'R Amplitude in Lead V6'
}

network = 'gan' 

count = 0
for parameter in parameters:
    count += 1
    mapped_param = param_mapping.get(parameter, parameter)  

    for lead in [1, 2]:  
        list_real = []
        list_generated = []
        list_I = []

        if count > 0 and count < 6:
            p = 'S_PeakAmpl_I'
            xlab = 'S'
        elif count > 5 and count < 11:
            p = 'STM_I'
            xlab = 'STM'
        elif count > 10 and count < 16:
            p = 'T_PeakAmpl_I'
            xlab = 'T'
        else:
            p = 'R_PeakAmpl_I'
            xlab = 'R'

        for ecg in range(8562, 9514):
            real = muse_df.loc[f'dataset_{ecg}', parameter]
            lead_I = muse_df.loc[f'dataset_{ecg}', p]

            if lead == 1:
                generated = muse_df.loc[f'gan_{lead}lead_{ecg}', parameter]
            else:
                generated = muse_df.loc[f'gan_{lead}leads_{ecg}', parameter]

            list_real.append(real)
            list_generated.append(generated)
            list_I.append(lead_I)

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
        font_axis_labels1.set_size(15)

        sns.scatterplot(x=list_I, y=list_real, color='steelblue', label='Real Values', s=20)
        sns.scatterplot(x=list_I, y=list_generated, color="rosybrown", label='Generated Values', s=20)

        r_real, _ = pearsonr(list_I, list_real)
        slope_real = r_real * np.std(list_real) / np.std(list_I)
        intercept_real = np.mean(list_real) - slope_real * np.mean(list_I)

        r_generated, _ = pearsonr(list_I, list_generated)
        slope_generated = r_generated * np.std(list_generated) / np.std(list_I)
        intercept_generated = np.mean(list_generated) - slope_generated * np.mean(list_I)

        xlim = plt.xlim()
        ylim = plt.ylim()
        x = np.linspace(xlim[0], xlim[1], 100)

        y_real = slope_real * x + intercept_real
        y_generated = slope_generated * x + intercept_generated

        sns.lineplot(x=x, y=y_real, color='midnightblue', label=f'R² for Real = {pow(r_real, 2):.2f}')
        sns.lineplot(x=x, y=y_generated, color="brown", label=f'R² for Generated = {pow(r_generated, 2):.2f}')

        plt.title(f'{mapped_param}', fontproperties=font_title)
        plt.xlabel(f'{xlab} amplitude in Lead II (µV)', fontproperties=font_axis_labels1)
        plt.ylabel(f'{xlab} amplitude in V3 (µV)', fontproperties=font_axis_labels1)
        plt.xticks(fontproperties=font_axis_labels)
        plt.yticks(fontproperties=font_axis_labels)

        plt.savefig(f'generated_plots/{parameter}_{network}_lead{lead}.pdf')
        plt.close(fig)
