from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MultipleLocator


def create_and_save_plot(leadI, leadsII_VIII, generated_leads_II_VIII, filename, file_extension='.pdf'):
    number_of_input_leads = leadI.shape[0]
    LINE_WIDTH = 0.3
    fig, axs = plt.subplots(4, 2, figsize=(18, 12))

    for i in range(4):
        for j in range(2):
            axs[i, j].yaxis.set_major_locator(MultipleLocator(0.5))
            axs[i, j].yaxis.set_minor_locator(AutoMinorLocator(4))
            axs[i, j].grid(which='major', color='#CCCCCC', linestyle='--', linewidth=LINE_WIDTH)
            axs[i, j].grid(which='minor', color='#CCCCCC', linestyle=':', linewidth=LINE_WIDTH / 2)

    for i in range(number_of_input_leads):
        axs[i, 0].plot(leadI[i], linewidth=LINE_WIDTH)
    for i in range(number_of_input_leads, 4):
        axs[i, 0].plot(leadsII_VIII[i - number_of_input_leads], linewidth=LINE_WIDTH)
        axs[i, 0].plot(generated_leads_II_VIII[i], linewidth=LINE_WIDTH)
    for i in range(4):
        axs[i, 1].plot(leadsII_VIII[i + 4 - number_of_input_leads], linewidth=LINE_WIDTH)
        axs[i, 1].plot(generated_leads_II_VIII[i + 4 - number_of_input_leads], linewidth=LINE_WIDTH)
    fig.savefig(Path(filename + file_extension))
    plt.close(fig)


def create_and_save_standardized_12_lead_ecg_plot(number_of_leads_as_input, leads, generated_leads, filename, plot_seconds=10, plot_columns=2, plot_range=1.8, file_extension='.pdf'):
    # code adapted from https://github.com/dy1901/ecg_plot
    def _ax_plot(ax, x, y1, y2, seconds=10, amplitude_ecg=1.8, time_ticks=0.2, plot_second_signal=True):
        ax.set_xticks(np.arange(0, 11, time_ticks))
        ax.set_yticks(np.arange(-ceil(amplitude_ecg), ceil(amplitude_ecg), 1.0))
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.set_ylim(-amplitude_ecg, amplitude_ecg)
        ax.set_xlim(0, seconds)
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
        ax.grid(which='minor', linestyle='-', linewidth='0.5', color=(1, 0.7, 0.7))
        ax.plot(x, y1, linewidth=1.5, color="black")
        if plot_second_signal:
            ax.plot(x, y2, linewidth=2, color="blue", linestyle='dotted')

    number_of_leads = 12
    voltage=20
    sample_rate=500
    speed=50
    lead_index = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    lead_order = list(range(0, number_of_leads))
    seconds = plot_seconds if plot_seconds is not None else len(leads[0]) / sample_rate
    plt.rc('axes', titleweight='bold')
    fig, ax = plt.subplots(
        ceil(number_of_leads/plot_columns), plot_columns,
        sharex=True, sharey=True,
        figsize=((speed/25.4)*seconds*plot_columns, (4.1*voltage/25.4)*number_of_leads/plot_columns))
    fig.subplots_adjust(hspace=0.01, wspace=0.02, left=0.01, right=0.98, bottom=0.06, top=0.95)

    step = 1.0 / sample_rate
    for i in range(0, number_of_leads):
        if(plot_columns == 1):
            t_ax = ax[i]
        else:
            t_ax = ax[i // plot_columns, i % plot_columns]
        t_lead = lead_order[i]
        t_ax.set_title(lead_index[t_lead], y=1.0, pad=-14)
        t_ax.tick_params(axis='x', rotation=90)
        plot_second_signal=generated_leads is not None and not (lead_index[t_lead] == "I" or (number_of_leads_as_input == 2 and lead_index[t_lead] == "II"))
        _ax_plot(t_ax, np.arange(0, len(leads[t_lead])*step, step), 
                 leads[t_lead], generated_leads[t_lead] if generated_leads is not None else None, 
                 seconds, amplitude_ecg=plot_range, plot_second_signal=plot_second_signal)
    fig.savefig(Path(filename + file_extension))
    plt.close(fig)


def createLossPlots(plot_filename, file_extension='.pdf', loss1 = None, loss2 = None, loss3 = None):
    plt.plot(loss1)
    plt.plot(loss2)
    if loss3:
        plt.plot(loss3)
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.savefig(Path(f"{plot_filename}{file_extension}"))
    plt.close()   