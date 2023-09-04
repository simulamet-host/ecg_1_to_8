import sys
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.stats import kendalltau, pearsonr, spearmanr

from datasets import EcgDataset
from networks import EcgUNetGenerator


@dataclass
class MetricsExtrema:
    greater_is_better: bool
    value: float = 0
    index: int = 0

    def __init__(self, greater_is_better: bool, minimum_value=-sys.float_info.max, maximum_value=sys.float_info.max) -> None:
        self.greater_is_better = greater_is_better
        if greater_is_better:
            self.value = minimum_value
        else:
            self.value = maximum_value

    def replace_if_needed(self, value, index) -> None:
        if self.greater_is_better:
            if self.value < value:
                self.value = value
                self.index = index
        else:
            if self.value > value:
                self.value = value
                self.index = index


@dataclass
class SignalMetrics:
    mse: list[float] = field(default_factory=list)
    mae: list[float] = field(default_factory=list)
    frechet: list[float] = field(default_factory=list)
    pearson: list[float] = field(default_factory=list)
    spearman: list[float] = field(default_factory=list)
    kendall: list[float] = field(default_factory=list)

    average_mse: float = 0
    average_mae: float = 0
    average_frechet: float = 0
    average_pearson: float = 0
    average_spearman: float = 0
    average_kendall: float = 0

    best_mse: list[MetricsExtrema] = field(default_factory=list)
    best_mae: list[MetricsExtrema] = field(default_factory=list)
    best_frechet: list[MetricsExtrema] = field(default_factory=list)
    best_pearson: list[MetricsExtrema] = field(default_factory=list)
    best_spearman: list[MetricsExtrema] = field(default_factory=list)
    best_kendall: list[MetricsExtrema] = field(default_factory=list)

    worst_mse: list[MetricsExtrema] = field(default_factory=list)
    worst_mae: list[MetricsExtrema] = field(default_factory=list)
    worst_frechet: list[MetricsExtrema] = field(default_factory=list)
    worst_pearson: list[MetricsExtrema] = field(default_factory=list)
    worst_spearman: list[MetricsExtrema] = field(default_factory=list)
    worst_kendall: list[MetricsExtrema] = field(default_factory=list)


def frechet_loss(signal1, signal2):
    return np.max(np.abs(np.subtract(signal1, signal2)))


def calculate_metrics(number_of_generated_leads: int, 
                      generator: EcgUNetGenerator, 
                      dataset: EcgDataset, 
                      dataloader: torch.utils.data.DataLoader,
                      target: str,
                      device) -> SignalMetrics:
    
    metrics: SignalMetrics = SignalMetrics()
    # plus best average and worst average
    for i in range(number_of_generated_leads + 1):
        if i < number_of_generated_leads:
            metrics.mse.append(0)
            metrics.mae.append(0)
            metrics.frechet.append(0)
            metrics.pearson.append(0)
            metrics.spearman.append(0)
            metrics.kendall.append(0)

        metrics.best_mse.append(MetricsExtrema(greater_is_better=False))
        metrics.best_mae.append(MetricsExtrema(greater_is_better=False))
        metrics.best_frechet.append(MetricsExtrema(greater_is_better=False))
        metrics.best_pearson.append(MetricsExtrema(greater_is_better=True))
        metrics.best_spearman.append(MetricsExtrema(greater_is_better=True))
        metrics.best_kendall.append(MetricsExtrema(greater_is_better=True))
        
        metrics.worst_mse.append(MetricsExtrema(greater_is_better=True))
        metrics.worst_mae.append(MetricsExtrema(greater_is_better=True))
        metrics.worst_frechet.append(MetricsExtrema(greater_is_better=True))
        metrics.worst_pearson.append(MetricsExtrema(greater_is_better=False))
        metrics.worst_spearman.append(MetricsExtrema(greater_is_better=False))
        metrics.worst_kendall.append(MetricsExtrema(greater_is_better=False))

    mse_loss = torch.nn.MSELoss()
    mae_loss = torch.nn.L1Loss()

    generator.eval()
    with torch.inference_mode():
        for batch, (source_leads, target_leads) in enumerate(dataloader):
            file_index = batch + dataset.get_start_index(target = target)
            generated_target_leads = generator(source_leads.to(device))[0].data.cpu()
            
            target_leads_milli = dataset.convert_to_millivolts(dataset.convert_output(target_leads[0]))
            generated_target_leads_milli = dataset.convert_to_millivolts(dataset.convert_output(generated_target_leads))

            sum_mse = 0
            sum_mae = 0
            sum_frechet = 0
            sum_pearson = 0
            sum_spearman = 0
            sum_kendall = 0
            for i in range(number_of_generated_leads):
                mse_result = mse_loss(target_leads_milli[i], generated_target_leads_milli[i]).item()
                mae_result = mae_loss(target_leads_milli[i], generated_target_leads_milli[i]).item()
                frechet_result = frechet_loss(target_leads_milli[i].numpy(), generated_target_leads_milli[i].numpy())
                pearson_result, p_pearson = pearsonr(target_leads_milli[i], generated_target_leads_milli[i])
                spearman_result, p_spearman = spearmanr(target_leads_milli[i], generated_target_leads_milli[i])
                kendall_result, p_kendall = kendalltau(target_leads_milli[i], generated_target_leads_milli[i])

                # this means that either the original or the generated signal has no variation (is constant)
                # we assume zero correlation in that case (not negative correlation)
                if np.isnan(pearson_result):
                    pearson_result = 0 
                if np.isnan(spearman_result):
                    spearman_result = 0
                if np.isnan(kendall_result):
                    kendall_result = 0

                metrics.best_mse[i].replace_if_needed(mse_result, file_index)
                metrics.best_mae[i].replace_if_needed(mae_result, file_index)
                metrics.best_frechet[i].replace_if_needed(frechet_result, file_index)
                metrics.best_pearson[i].replace_if_needed(pearson_result, file_index)
                metrics.best_spearman[i].replace_if_needed(spearman_result, file_index)
                metrics.best_kendall[i].replace_if_needed(kendall_result, file_index)
                metrics.worst_mse[i].replace_if_needed(mse_result, file_index)
                metrics.worst_mae[i].replace_if_needed(mae_result, file_index)
                metrics.worst_frechet[i].replace_if_needed(frechet_result, file_index)
                metrics.worst_pearson[i].replace_if_needed(pearson_result, file_index)
                metrics.worst_spearman[i].replace_if_needed(spearman_result, file_index)
                metrics.worst_kendall[i].replace_if_needed(kendall_result, file_index)

                sum_mse += mse_result
                sum_mae += mae_result
                sum_frechet += frechet_result
                sum_pearson += pearson_result
                sum_spearman += spearman_result
                sum_kendall += kendall_result
                metrics.mse[i] += mse_result
                metrics.mae[i] += mae_result
                metrics.frechet[i] += frechet_result
                metrics.pearson[i] += pearson_result
                metrics.spearman[i] += spearman_result
                metrics.kendall[i] += kendall_result

            average_mse = sum_mse / number_of_generated_leads
            average_mae = sum_mae / number_of_generated_leads
            average_frechet = sum_frechet / number_of_generated_leads
            average_pearson = sum_pearson / number_of_generated_leads
            average_spearman = sum_spearman / number_of_generated_leads
            average_kendall = sum_kendall / number_of_generated_leads
            metrics.best_mse[number_of_generated_leads].replace_if_needed(average_mse, file_index)
            metrics.best_mae[number_of_generated_leads].replace_if_needed(average_mae, file_index)
            metrics.best_frechet[number_of_generated_leads].replace_if_needed(average_frechet, file_index)
            metrics.best_pearson[number_of_generated_leads].replace_if_needed(average_pearson, file_index)
            metrics.best_spearman[number_of_generated_leads].replace_if_needed(average_spearman, file_index)
            metrics.best_kendall[number_of_generated_leads].replace_if_needed(average_kendall, file_index)
            metrics.worst_mse[number_of_generated_leads].replace_if_needed(average_mse, file_index)
            metrics.worst_mae[number_of_generated_leads].replace_if_needed(average_mae, file_index)
            metrics.worst_frechet[number_of_generated_leads].replace_if_needed(average_frechet, file_index)
            metrics.worst_pearson[number_of_generated_leads].replace_if_needed(average_pearson, file_index)
            metrics.worst_spearman[number_of_generated_leads].replace_if_needed(average_spearman, file_index)
            metrics.worst_kendall[number_of_generated_leads].replace_if_needed(average_kendall, file_index)

        for i in range(number_of_generated_leads):
            metrics.mse[i] /= len(dataset)
            metrics.mae[i] /= len(dataset)
            metrics.frechet[i] /= len(dataset)
            metrics.pearson[i] /= len(dataset)
            metrics.spearman[i] /= len(dataset)
            metrics.kendall[i] /= len(dataset)

        metrics.average_mse = np.average(metrics.mse)
        metrics.average_mae = np.average(metrics.mae)
        metrics.average_frechet = np.average(metrics.frechet)
        metrics.average_pearson = np.average(metrics.pearson)
        metrics.average_spearman = np.average(metrics.spearman)
        metrics.average_kendall = np.average(metrics.kendall)

    return metrics
