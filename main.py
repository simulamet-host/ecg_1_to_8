import argparse
import random
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
import torch

import wb
from datasets import EcgDataset, get_dataloader
from networks import EcgGanDiscriminator, EcgUNetGenerator
from plots import create_and_save_plot, create_and_save_standardized_12_lead_ecg_plot, createLossPlots
from signal_metrics import SignalMetrics, calculate_metrics

USE_WEIGHTS_AND_BIASES = False
WANDB_KEY = ''

ACTION = 'train'                                 # train | test | generate_outputs | generate

############# CONFIGURATION ###############
train_configuration = {
    'DATASET_OPTION': 'PTB',                     # synthetic | PTB | ['synthetic', 'PTB']
    'INPUT_LEADS': 1,
    'NETWORK_OPTION': 'GAN',                     # GAN | UNET
    'LEARNING_RATE': 0.0001,
    'MODEL_SIZE': 32,
    'BATCH_SIZE': 32,
    'EPOCHS': 500,
    'DISCRIMINATOR_PATCH_SIZE': 1000,
    'SAVE_EVERY_N_EPOCHS': 5
}

test_configuration = {
    'DATASET_OPTION': 'PTB',
    'INPUT_LEADS': 1,
    'NETWORK_OPTION': 'GAN',
    'MODEL_SIZE': 32,
    'SAVED_MODEL': "test_models/gan_generator_epoch_120",
    'OUTPUTS_FOLDER': "test_models",
    'INPUT_PATH': "test_models/example_input.csv",
    'GENERATE_PLOTS': False,
    'PLOT_SECONDS': 10,
    'PLOT_COLUMNS': 2,
    'PLOT_RANGE': 1.8,
    'GENERATE_CSV': False,
    'NORMALIZE_FACTOR': 8,
    'GENERATE_STEVEN_MODEL': False,
    'SPECIFIC_INDEXES': None,
    'LIMIT': None
}
############################################

PLOTS_FOLDER = "plots"
SAVED_MODELS_FOLDER_UNET = "saved_models_unet"
SAVED_MODELS_FOLDER_GAN = "saved_models_gan"
NUMBER_OF_SAMPLES_IN_SIGNAL = 5000
NUMBER_OF_LEADS = 8

def create_folder_if_not_exists(folder: str):
    if not(Path(folder).is_dir()):
        Path(folder).mkdir(parents=True, exist_ok=True)


def fix_seed():
    # torch.use_deterministic_algorithms(True)
    # import os
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    random_seed = 123
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 


def create_8_lead_ecg(source_leads, target_leads) -> np.array:
    return np.array([*source_leads, *target_leads])


def compute_leadIII_aVR_aVL_aVF(source_leads, target_leads):
    # single lead as input or two leads as input
    if source_leads.shape[0] == 1:
        lead_III_value = target_leads[0] - source_leads[0] # (lead II value) - (lead I value)
        lead_aVR_value = -(source_leads[0] + target_leads[0]) / 2 # -0.5*(lead I value + lead II value)
        lead_aVL_value = source_leads[0] - target_leads[0] / 2 # lead I value - 0.5 * lead II value
        lead_aVF_value = target_leads[0] - source_leads[0] / 2 # lead II value - 0.5 * lead I value
    else:
        lead_III_value = source_leads[1] - source_leads[0]
        lead_aVR_value = -(source_leads[0] + source_leads[1]) / 2
        lead_aVL_value = source_leads[0] - source_leads[1] / 2
        lead_aVF_value = source_leads[1] - source_leads[0] / 2
    return lead_III_value, lead_aVR_value, lead_aVL_value, lead_aVF_value


def create_12_lead_ecg(source_leads, target_leads) -> np.array:
    lead_III_value, lead_aVR_value, lead_aVL_value, lead_aVF_value = compute_leadIII_aVR_aVL_aVF(source_leads, target_leads)
    
    # single lead as input or two leads as input
    if source_leads.shape[0] == 1:
        ecg_values = np.array([source_leads[0], target_leads[0], lead_III_value, lead_aVR_value, lead_aVL_value, lead_aVF_value,
                            target_leads[1], target_leads[2], target_leads[3], target_leads[4], target_leads[5], target_leads[6]])
    else:
        ecg_values = np.array([source_leads[0], source_leads[1], lead_III_value, lead_aVR_value, lead_aVL_value, lead_aVF_value,
                            target_leads[0], target_leads[1], target_leads[2], target_leads[3], target_leads[4], target_leads[5]])
    return ecg_values


def create_and_save_12_lead_csv(source_leads, target_leads, filename):
    ecg_values = create_12_lead_ecg(source_leads, target_leads)
    pd.DataFrame(ecg_values).transpose().to_csv(filename, index=False, header=False)


def create_and_save_8_lead_csv(source_leads, target_leads, filename):
    ecg_values = create_8_lead_ecg(source_leads, target_leads)
    pd.DataFrame(ecg_values).transpose().to_csv(filename, index=False, header=False)


def createOutputPlots(number_of_leads_as_input: int, 
                      plot_filename: str, 
                      dataset: EcgDataset, 
                      generator: EcgUNetGenerator, 
                      sample_source: Literal["train", "test", "validation"] = "validation", 
                      random_sample: bool = False, 
                      file_extension: str='.pdf'):
    with torch.inference_mode():
        leadI, leadsII_VIII = dataset.get_sample(source=sample_source, random_sample = random_sample)
        leadI_reshaped = torch.reshape(leadI, (1, number_of_leads_as_input, NUMBER_OF_SAMPLES_IN_SIGNAL))

        generated_leadsII_VIII = generator(leadI_reshaped.to(device)).data.cpu()

        leadI_milli = dataset.convert_to_millivolts(dataset.convert_output(leadI_reshaped[0]))
        leadsII_VIII_milli = dataset.convert_to_millivolts(dataset.convert_output(leadsII_VIII))
        generated_leadsII_VIII_milli = dataset.convert_to_millivolts(dataset.convert_output(generated_leadsII_VIII[0]))

        create_and_save_plot(leadI_milli, leadsII_VIII_milli, generated_leadsII_VIII_milli, plot_filename, file_extension)


def create_and_save_12_lead_ecg_plot(number_of_leads_as_input: int, source_leads, target_leads, generated_target_leads, filename, plot_seconds=10, plot_columns=2, plot_range=1.8):
    ecg_values = create_12_lead_ecg(source_leads, target_leads)
    ecg_values_generated = create_12_lead_ecg(source_leads, generated_target_leads) if generated_target_leads is not None else None
    create_and_save_standardized_12_lead_ecg_plot(number_of_leads_as_input, ecg_values, ecg_values_generated, filename, plot_seconds, plot_columns, plot_range)


def test(number_of_leads_as_input: int,
         generator_path: str,
         model_size: int,
         test_dataset: EcgDataset,
         test_dataloader: torch.utils.data.DataLoader,
         target: str = 'test'):
    number_of_generated_leads = NUMBER_OF_LEADS-number_of_leads_as_input
    generator = EcgUNetGenerator(num_input_channels=number_of_leads_as_input,
                                 num_output_channels=number_of_generated_leads,
                                 model_size=model_size)
    generator.load_state_dict(torch.load(Path(generator_path), map_location=torch.device(device)))
    generator = generator.to(device)
    metrics = calculate_metrics(number_of_generated_leads, generator, test_dataset, test_dataloader, target, device)
    pprint(metrics)


def validate_generator(number_of_leads_as_input: int, 
                       generator: EcgUNetGenerator, 
                       validation_dataset: EcgDataset, 
                       validation_dataloader: torch.utils.data.DataLoader) -> SignalMetrics:
    number_of_generated_leads = NUMBER_OF_LEADS-number_of_leads_as_input
    return calculate_metrics(number_of_generated_leads, 
                             generator, 
                             validation_dataset, 
                             validation_dataloader, 
                             target='validation',
                             device=device)


def train_gan(number_of_leads_as_input: int, 
              model_size: int, 
              learning_rate: float, 
              number_of_epochs: int, 
              batch_size: int,
              save_every_n_epochs: int, 
              discriminator_patch_size: int,
              train_dataset: EcgDataset, train_dataloader: torch.utils.data.DataLoader,
              validation_dataset: EcgDataset, validation_dataloader: torch.utils.data.DataLoader):
    TRAIN_GENERATOR_EVERY_N_BATCHES = 5
    generator = EcgUNetGenerator(num_input_channels=number_of_leads_as_input,
                                 num_output_channels=NUMBER_OF_LEADS-number_of_leads_as_input,
                                 model_size=model_size).to(device)
    discriminator = EcgGanDiscriminator(num_channels=NUMBER_OF_LEADS).to(device)

    class GANLoss(torch.nn.Module):
        """Define different GAN objectives.

        The GANLoss class abstracts away the need to create the target label tensor
        that has the same size as the input.
        """
        def __init__(self, gan_mode: Literal['MSE', 'BCE', 'none'], target_real_label=1.0, target_fake_label=0.0):
            """ Initialize the GANLoss class.

            Parameters:
                gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
                target_real_label (float) - - label for a real image
                target_fake_label (float) - - label of a fake image

            Note: Do not use sigmoid as the last layer of Discriminator.
            LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
            """
            super(GANLoss, self).__init__()
            self.register_buffer('real_label', torch.tensor(target_real_label))
            self.register_buffer('fake_label', torch.tensor(target_fake_label))
            self.gan_mode = gan_mode
            if gan_mode == 'MSE':
                self.loss = torch.nn.MSELoss()
            elif gan_mode == 'BCE':
                self.loss = torch.nn.BCEWithLogitsLoss() # This loss combines a Sigmoid layer and the BCELoss in one single class
            elif gan_mode == 'none':
                self.loss = None

        def get_target_tensor(self, prediction, target_is_real):
            """Create label tensors with the same size as the input.

            Parameters:
                prediction (tensor) - - tpyically the prediction from a discriminator
                target_is_real (bool) - - if the ground truth label is for real images or fake images

            Returns:
                A label tensor filled with ground truth label, and with the size of the input
            """

            if target_is_real:
                target_tensor = self.real_label
            else:
                target_tensor = self.fake_label
            return target_tensor.expand_as(prediction)

        def __call__(self, prediction, target_is_real):
            """Calculate loss given Discriminator's output and grount truth labels.

            Parameters:
                prediction (tensor) - - tpyically the prediction output from a discriminator
                target_is_real (bool) - - if the ground truth label is for real images or fake images

            Returns:
                the calculated loss.
            """
            if self.gan_mode in ['MSE', 'BCE']:
                target_tensor = self.get_target_tensor(prediction, target_is_real)
                loss = self.loss(prediction, target_tensor)
            elif self.gan_mode == 'none':
                if target_is_real:
                    loss = -prediction.mean()
                else:
                    loss = prediction.mean()
            return loss
    
    optimizerG = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    gan_loss = GANLoss("MSE").to(device)  # MSE | BCE | none
    generator_objective_loss = torch.nn.L1Loss(reduction='mean')  # L1Loss MSELoss KLDivLoss HuberLoss SmoothL1Loss
    generator_objective_loss_coef = 200 # 100.0

    # Adapted from https://github.com/caogang/wgan-gp/blob/master/gan_toy.py
    def calc_gradient_penalty(discriminator, real_data, generated_data, batch_size, lmbda = 10.0):
        # Compute interpolation factors
        alpha = torch.rand(batch_size, 1, 1).to(device)
        alpha = alpha.expand(real_data.size())
        interpolated = alpha * real_data + (1 - alpha) * generated_data
        interpolated = torch.autograd.Variable(interpolated, requires_grad=True)

        # Evaluate discriminator
        discriminator_prediction_on_interpolated = discriminator(interpolated)
        gradient_outputs = torch.ones(discriminator_prediction_on_interpolated.size()).to(device)

        # Obtain gradients of the discriminator with respect to the inputs
        gradients = torch.autograd.grad(outputs=discriminator_prediction_on_interpolated, 
                                        inputs=interpolated, 
                                        grad_outputs=gradient_outputs,
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        # Compute MSE between 1.0 and the gradient of the norm penalty to make discriminator to be a 1-Lipschitz function.
        gradient_penalty = lmbda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    generator_loss_plot = []
    discriminator_loss_plot = []
    validation_loss_plot = []
    for epoch in range(1, number_of_epochs + 1):
        train_generator_flag = False
        discriminator_loss_list = []
        discriminator_loss_on_generated_list = []
        generator_loss_list = []
        generator_objective_loss_list = []

        generator.train()
        for batch, (source_leads, target_leads) in enumerate(train_dataloader, 0):
            if not USE_WEIGHTS_AND_BIASES and batch % 20 == 0:
                print(f'Batch: {batch}, time {datetime.now()}')

            # Train generator only after every n-th batch
            train_generator_flag = (batch + 1) % TRAIN_GENERATOR_EVERY_N_BATCHES == 0

            # Set Discriminator parameters to require gradients.
            for p in discriminator.parameters():
                p.requires_grad = True

            source_leads = source_leads.to(device)
            target_leads = target_leads.to(device)
            ##########################
            # (1) Train Discriminator
            ##########################
            optimizerD.zero_grad()

            # a) Real
            patch_start = random.randint(0, NUMBER_OF_SAMPLES_IN_SIGNAL - discriminator_patch_size)
            source_leads_patch = source_leads[:, :, patch_start:patch_start + discriminator_patch_size]
            target_leads_patch = target_leads[:, :, patch_start:patch_start + discriminator_patch_size]
            real_input_real_output = torch.cat((source_leads_patch, target_leads_patch), dim=1)
            discriminator_prediction_on_real = discriminator(real_input_real_output)
            discriminator_loss_on_real = gan_loss(prediction=discriminator_prediction_on_real, target_is_real=True)
        
            # b) Generated
            source_leads_variable = torch.autograd.Variable(source_leads, requires_grad=False)
            generated_output_variable = torch.autograd.Variable(generator(source_leads_variable).data)
            generated_output_variable_patch = generated_output_variable[:, :, patch_start:patch_start + discriminator_patch_size]
            real_input_generated_output_variable = torch.cat((source_leads_patch, generated_output_variable_patch), dim=1)
            discriminator_prediction_on_generated = discriminator(real_input_generated_output_variable)

            discriminator_loss_on_generated = gan_loss(prediction=discriminator_prediction_on_generated, target_is_real=False)

            # combine loss and calculate gradients
            combined_discriminator_loss = discriminator_loss_on_generated + discriminator_loss_on_real
            combined_discriminator_loss.backward()

            # c) compute gradient penalty and backprop
            gradient_penalty = calc_gradient_penalty(discriminator, real_input_real_output, real_input_generated_output_variable.data, batch_size)
            gradient_penalty.backward()
            optimizerD.step()

            discriminator_loss_list.append(combined_discriminator_loss.data.cpu())
            discriminator_loss_on_generated_list.append(discriminator_loss_on_generated.data.cpu())
       
            ####################################
            # (2) Train Generator
            ####################################
            if train_generator_flag:
                # Prevent discriminator update
                for p in discriminator.parameters():
                    p.requires_grad = False

                # Reset generator gradients
                optimizerG.zero_grad()

                # First, G(A) should fake the discriminator
                source_leads_variable = torch.autograd.Variable(source_leads, requires_grad=False)
                generated_output = generator(source_leads_variable)
                real_input_generated_output_variable = torch.cat((source_leads, generated_output), dim=1)
                real_input_generated_output_variable_patch = real_input_generated_output_variable[:,:, patch_start:patch_start + discriminator_patch_size]

                discriminator_prediction_on_generated = discriminator(real_input_generated_output_variable_patch)
                generator_gan_loss = gan_loss(prediction=discriminator_prediction_on_generated, target_is_real=True)
                weighted_generator_objective_loss = generator_objective_loss(generated_output, target_leads) * generator_objective_loss_coef
                combined_generator_loss = generator_gan_loss + weighted_generator_objective_loss
                combined_generator_loss.backward()

                optimizerG.step()

                generator_loss_list.append(combined_generator_loss.data.cpu())
                generator_objective_loss_list.append(weighted_generator_objective_loss.data.cpu())
        
        validation_metrics = validate_generator(number_of_leads_as_input, generator, validation_dataset, validation_dataloader)
        discriminator_loss_average = np.average(discriminator_loss_list)
        discriminator_loss_on_generated_average = np.average(discriminator_loss_on_generated_list)
        generator_loss_average = np.average(generator_loss_list)
        generator_objective_loss_average = np.average(generator_objective_loss_list)
        
        validation_loss_plot.append(validation_metrics.average_mse)
        discriminator_loss_plot.append(discriminator_loss_average)
        generator_loss_plot.append(generator_loss_average)

        # print stats
        if USE_WEIGHTS_AND_BIASES:
            wandb.log({"Discriminator_loss": discriminator_loss_average,
                       "Generator_loss": generator_loss_average,
                       "Validation_MSE": validation_metrics.average_mse,
                       "Validation_MAE": validation_metrics.average_mae,
                       "Validation_frechet": validation_metrics.average_frechet,
                       "Validation_pearson": validation_metrics.average_pearson,
                       "Validation_spearman": validation_metrics.average_spearman,
                       "Validation_kendall": validation_metrics.average_kendall,
                       })
        else:
            print(f"Epoch:{epoch} "
                  f"G_cost:{generator_loss_average:.4f}  G_obj_cost:{generator_objective_loss_average:.4f} "
                  f"D_cost:{discriminator_loss_average:.4f}  D_cost_fake:{discriminator_loss_on_generated_average:.4f} "
                  f"Validation_MSE:{validation_metrics.average_mse:.4f}  "
                  f"Validation_MAE:{validation_metrics.average_mae:.4f}  "
                  f"Validation_frechet:{validation_metrics.average_frechet:.4f}  "
                  f"Validation_pearson:{validation_metrics.average_pearson:.4f}  "
                  f"Validation_spearman:{validation_metrics.average_spearman:.4f}  "
                  f"Validation_kendall:{validation_metrics.average_kendall:.4f}")


        # save plots
        plot_filename = f"{PLOTS_FOLDER}/ecg{epoch}"
        if USE_WEIGHTS_AND_BIASES:
            createOutputPlots(number_of_leads_as_input, plot_filename, validation_dataset, generator, random_sample=True, file_extension=".png")
            wandb.log({"ECG": wandb.Image(plot_filename + ".png")})
        else:
            createOutputPlots(number_of_leads_as_input, plot_filename, validation_dataset, generator, random_sample=True)
            plot_train_filename = f"{PLOTS_FOLDER}/ecg{epoch}_train"
            createOutputPlots(number_of_leads_as_input, plot_train_filename, train_dataset, generator, sample_source="train", random_sample=True)
            plot_losses_filename = f"{PLOTS_FOLDER}/losses"
            createLossPlots(plot_losses_filename, loss1=generator_loss_plot, loss2=discriminator_loss_plot, loss3=validation_loss_plot)
    
        # save models
        if epoch % save_every_n_epochs == 0:
            if USE_WEIGHTS_AND_BIASES:
                generator_model_filename=f"{SAVED_MODELS_FOLDER_GAN}/gan_generator"
                discriminator_model_filename=f"{SAVED_MODELS_FOLDER_GAN}/gan_discriminator"
                torch.save(generator.state_dict(), generator_model_filename)
                torch.save(discriminator.state_dict(), discriminator_model_filename)
                wandb.log_artifact(generator_model_filename, name=f'generator_epoch_{epoch}', type='Model') 
                wandb.log_artifact(discriminator_model_filename, name=f'discriminator_epoch_{epoch}', type='Model') 
            else:
                generator_model_filename=f"{SAVED_MODELS_FOLDER_GAN}/gan_generator_epoch{epoch}.pt"
                discriminator_model_filename=f"{SAVED_MODELS_FOLDER_GAN}/gan_discriminator_epoch{epoch}.pt"
                torch.save(generator.state_dict(), generator_model_filename)
                torch.save(discriminator.state_dict(), discriminator_model_filename)


def train_unet(number_of_leads_as_input: int,
               model_size: int,
               learning_rate: float,
               number_of_epochs: int,
               save_every_n_epochs: int,
               train_dataset: EcgDataset, train_dataloader: torch.utils.data.DataLoader,
               validation_dataset: EcgDataset, validation_dataloader: torch.utils.data.DataLoader):
    generator = EcgUNetGenerator(num_input_channels=number_of_leads_as_input,
                                 num_output_channels=NUMBER_OF_LEADS-number_of_leads_as_input,
                                 model_size=model_size).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    train_loss_plot = []
    validation_loss_plot = []

    for epoch in range(1, number_of_epochs + 1):
        generator.train()
        train_loss_average = 0
        for batch, (source_leads, target_leads) in enumerate(train_dataloader, 0):
            if not USE_WEIGHTS_AND_BIASES and batch % 20 == 0: 
                print(f'Batch: {batch}, time {datetime.now()}')

            source_leads = source_leads.to(device)
            target_leads = target_leads.to(device)

            output = generator(source_leads)
            train_criterion = criterion(train_dataset.convert_to_millivolts(train_dataset.convert_output(output)), 
                                        train_dataset.convert_to_millivolts(train_dataset.convert_output(target_leads)))
            train_loss_average += train_criterion.data.cpu()
            optimizer.zero_grad()
            train_criterion.backward()
            optimizer.step()

        train_loss_average /= len(train_dataloader)
        validation_metrics = validate_generator(number_of_leads_as_input, generator, validation_dataset, validation_dataloader)
        
        train_loss_plot.append(train_loss_average)
        validation_loss_plot.append(validation_metrics.average_mse)

        # print stats
        if USE_WEIGHTS_AND_BIASES:
            wandb.log({"Train_loss": train_loss_average,
                       "Validation_MSE": validation_metrics.average_mse,
                       "Validation_MAE": validation_metrics.average_mae,
                       "Validation_frechet": validation_metrics.average_frechet,
                       "Validation_pearson": validation_metrics.average_pearson,
                       "Validation_spearman": validation_metrics.average_spearman,
                       "Validation_kendall": validation_metrics.average_kendall,
                       })
        else:
            print(f"Epoch:{epoch} "
                  f"train_loss:{train_loss_average:.4f}  "
                  f"Validation_MSE:{validation_metrics.average_mse:.4f}  "
                  f"Validation_MAE:{validation_metrics.average_mae:.4f}  "
                  f"Validation_frechet:{validation_metrics.average_frechet:.4f}  "
                  f"Validation_pearson:{validation_metrics.average_pearson:.4f}  "
                  f"Validation_spearman:{validation_metrics.average_spearman:.4f}  "
                  f"Validation_kendall:{validation_metrics.average_kendall:.4f}")

        # save plots
        plot_filename = f"{PLOTS_FOLDER}/ecg{epoch}"
        if USE_WEIGHTS_AND_BIASES:
            createOutputPlots(number_of_leads_as_input, plot_filename, validation_dataset, generator, random_sample=True, file_extension=".png")
            wandb.log({"ECG": wandb.Image(plot_filename + ".png")})
        else:
            createOutputPlots(number_of_leads_as_input, plot_filename, validation_dataset, generator, random_sample=True)
            plot_train_filename = f"{PLOTS_FOLDER}/ecg{epoch}_train"
            createOutputPlots(number_of_leads_as_input, plot_train_filename, train_dataset, generator, sample_source="train", random_sample=True)
            plot_losses_filename = f"{PLOTS_FOLDER}/losses"
            createLossPlots(plot_losses_filename, loss1=train_loss_plot, loss2=validation_loss_plot)

        # save models
        if epoch % save_every_n_epochs == 0:
            if USE_WEIGHTS_AND_BIASES:
                model_filename=f"{SAVED_MODELS_FOLDER_UNET}/model"
                torch.save(generator.state_dict(), model_filename)
                wandb.log_artifact(model_filename, name=f'model_epoch_{epoch}', type='Model') 
            else:
                model_filename=f"{SAVED_MODELS_FOLDER_UNET}/model_epoch{epoch}.pt"
                torch.save(generator.state_dict(), model_filename)


def ensure_output_folders_exist(output_folder: str,
                                generate_plots: bool = True,
                                generate_csv: bool = True,
                                generate_steven_model: bool = False) -> Tuple[str, str]:
    plots_folder = f"{output_folder}/plots"
    csv_folder = f"{output_folder}/csv"
    if generate_plots:
        create_folder_if_not_exists(plots_folder)
    if generate_csv or generate_steven_model:
        create_folder_if_not_exists(csv_folder)
    return plots_folder, csv_folder


def generate_outputs(number_of_leads_as_input: int,
                     input_path: str,
                     generator_path: str,
                     model_size: int,
                     output_folder: str,
                     generate_plots: bool = True,
                     plot_seconds: float = 10,
                     plot_columns: int = 2,
                     plot_range: float = 1.8,
                     generate_csv: bool = True,
                     normalize_factor: float = 8):
    plots_folder, csv_folder = ensure_output_folders_exist(output_folder, generate_plots, generate_csv)
    
    generator = EcgUNetGenerator(num_input_channels=number_of_leads_as_input,
                                 num_output_channels=NUMBER_OF_LEADS-number_of_leads_as_input,
                                 model_size=model_size)
    generator.load_state_dict(torch.load(Path(generator_path), map_location=torch.device(device)))
    generator = generator.to(device)
    generator.eval()

    temp_df = pd.read_csv(input_path, header=None, index_col=False, dtype=float)
    source_leads = torch.tensor(np.clip(temp_df.iloc[:, :number_of_leads_as_input].values / normalize_factor, a_min=-1, a_max=1), dtype=torch.float32).t().unsqueeze(0)
    source_leads_array = np.array(source_leads[0] * normalize_factor, dtype=float)
    
    with torch.inference_mode():
        generated_target_leads = generator(source_leads.to(device))[0].data.cpu()
        generated_target_leads_array = np.array(generated_target_leads * normalize_factor, dtype=float)
        filename = Path(input_path).stem
        if generate_plots:
            plot_filename = f"{plots_folder}/{filename}_ecg_plot"
            create_and_save_12_lead_ecg_plot(number_of_leads_as_input, source_leads_array, generated_target_leads_array, None, plot_filename, plot_seconds, plot_columns, plot_range)
        if generate_csv:
            csv_12_lead_filename = f"{csv_folder}/{filename}_generated.csv"
            create_and_save_12_lead_csv(source_leads_array, generated_target_leads_array, csv_12_lead_filename)


def generate_outputs_from_test_data(number_of_leads_as_input: int,
                                    generator_path: str,
                                    model_size: int,
                                    test_dataset: EcgDataset,
                                    test_dataloader: torch.utils.data.DataLoader,
                                    output_folder: str,
                                    generate_plots: bool = True,
                                    plot_seconds: float = 10,
                                    plot_columns: int = 2,
                                    plot_range: float = 1.8,
                                    generate_csv: bool = True,
                                    generate_steven_model: bool = False,
                                    limit: int = None,
                                    specific_indexes: List[int] = None):
    plots_folder, csv_folder = ensure_output_folders_exist(output_folder, generate_plots, generate_csv, generate_steven_model)

    generator = EcgUNetGenerator(num_input_channels=number_of_leads_as_input,
                                 num_output_channels=NUMBER_OF_LEADS-number_of_leads_as_input,
                                 model_size=model_size)
    generator.load_state_dict(torch.load(Path(generator_path), map_location=torch.device(device)))
    generator = generator.to(device)
    generator.eval()

    if generate_steven_model:
        from steven_model import generate_signal_metric
        steven_model_names = ["HR rh", "QT rh", "Rpeak rh", "STJ rh", "Tpeak rh"]
        steven_model_names_generated = [name + " generated" for name in steven_model_names]
        steven_model_csv_filename = f"{csv_folder}/steven_model.csv"
        steven_model_df = pd.DataFrame(columns=steven_model_names + steven_model_names_generated)

    with torch.inference_mode():
        for batch, (source_leads, target_leads) in enumerate(test_dataloader):
            if batch == limit:
                break
            if specific_indexes and not (batch + test_dataset.get_start_index(target="test")) in specific_indexes:
                continue
            print(f"Generating index {batch}, time {datetime.now()}")
            generated_target_leads = generator(source_leads.to(device))[0].data.cpu()

            source_leads_milli = test_dataset.convert_to_millivolts(test_dataset.convert_output(np.array(source_leads[0], dtype=float)))
            target_leads_milli = test_dataset.convert_to_millivolts(test_dataset.convert_output(np.array(target_leads[0], dtype=float)))
            generated_target_leads_milli = test_dataset.convert_to_millivolts(test_dataset.convert_output(np.array(generated_target_leads, dtype=float)))

            file_index = batch + test_dataset.get_start_index(target="test")

            if generate_plots:
                # create the 12 lead ecg plot
                plot_filename = f"{plots_folder}/ecg_plot{file_index}"
                create_and_save_12_lead_ecg_plot(number_of_leads_as_input, source_leads_milli, target_leads_milli, generated_target_leads_milli, plot_filename, plot_seconds, plot_columns, plot_range)
                
                plot_filename = f"{plots_folder}/plot{file_index}"
                create_and_save_plot(source_leads_milli, target_leads_milli, generated_target_leads_milli, plot_filename)

            if generate_csv:
                # create 12 lead csv file in millivolts
                csv_12_lead_filename = f"{csv_folder}/generated_12_lead_{file_index}.csv"
                create_and_save_12_lead_csv(source_leads_milli, generated_target_leads_milli, csv_12_lead_filename)

                # create 8 lead csv file in microvolts
                source_leads_micro = np.rint(source_leads_milli * 1000).astype(int)
                target_leads_micro = np.rint(target_leads_milli * 1000).astype(int)
                generated_target_leads_micro = np.rint(generated_target_leads_milli * 1000).astype(int)
                csv_filename = f"{csv_folder}/generated_{file_index}.csv"
                csv_filename_real = f"{csv_folder}/dataset_{file_index}.csv"
                create_and_save_8_lead_csv(source_leads_micro, generated_target_leads_micro, csv_filename)
                create_and_save_8_lead_csv(source_leads_micro, target_leads_micro, csv_filename_real)

            if generate_steven_model:
                source_leads_micro = source_leads_milli * 1000
                target_leads_micro = target_leads_milli * 1000
                generated_target_leads_micro = generated_target_leads_milli * 1000
                real_8_leads = np.expand_dims(create_8_lead_ecg(source_leads_micro, target_leads_micro).transpose(), axis=0)
                generated_8_leads = np.expand_dims(create_8_lead_ecg(source_leads_micro, generated_target_leads_micro).transpose(), axis=0)
                steven_model_results = []
                for steven_model_name in steven_model_names:
                    steven_model_results.append(generate_signal_metric(steven_model_name, submodel_index=0, input_8_lead_signal=real_8_leads))
                for steven_model_name in steven_model_names:
                    steven_model_results.append(generate_signal_metric(steven_model_name, submodel_index=0, input_8_lead_signal=generated_8_leads))
                steven_model_df.loc[len(steven_model_df)] = steven_model_results
    if generate_steven_model:
        steven_model_df.to_csv(steven_model_csv_filename, index=False, header=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=__name__, description='ECG 1-to-8.')
    parser.add_argument('--action', type=str, choices=["train", "test", "generate_outputs", "generate"], default=ACTION)
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--wandb-key', type=str, default=WANDB_KEY)
    parser.add_argument('--dataset', type=str, choices=["synthetic", "PTB", "PTB_pathologic"], action='append')
    parser.add_argument('--input-leads', type=int, default=train_configuration['INPUT_LEADS'])
    parser.add_argument('--network', type=str, choices=["GAN", "UNET"], default=train_configuration['NETWORK_OPTION'])
    parser.add_argument('--learning-rate', type=float, default=train_configuration['LEARNING_RATE'])
    parser.add_argument('--model-size', type=int, default=train_configuration['MODEL_SIZE'])
    parser.add_argument('--batch-size', type=int, default=train_configuration['BATCH_SIZE'])
    parser.add_argument('--epochs', type=int, default=train_configuration['EPOCHS'])
    parser.add_argument('--save-every', type=int, default=train_configuration['SAVE_EVERY_N_EPOCHS'])
    parser.add_argument('--patch-size', type=int, default=train_configuration['DISCRIMINATOR_PATCH_SIZE'])
    parser.add_argument('--saved-model', type=str, default=test_configuration['SAVED_MODEL'])
    parser.add_argument('--outputs-folder', type=str, default=test_configuration['OUTPUTS_FOLDER'])
    parser.add_argument('--input', type=str, default=test_configuration['INPUT_PATH'])
    parser.add_argument('--plots', action='store_true', default=test_configuration['GENERATE_PLOTS'])
    parser.add_argument('--seconds', type=float, default=test_configuration['PLOT_SECONDS'])
    parser.add_argument('--columns', type=int, default=test_configuration['PLOT_COLUMNS'])
    parser.add_argument('--range', type=float, default=test_configuration['PLOT_RANGE'])
    parser.add_argument('--csv', action='store_true', default=test_configuration['GENERATE_CSV'])
    parser.add_argument('--normalize-factor', type=float, default=test_configuration['NORMALIZE_FACTOR'])
    parser.add_argument('--steven-model', action='store_true', default=False)
    parser.add_argument('--index', type=int, action='append', default=None)
    parser.add_argument('--limit', type=int, default=test_configuration['LIMIT'])
    options, unknown = parser.parse_known_args()
    print(options)
    
    ACTION = options.action

    if ACTION == 'train':
        USE_WEIGHTS_AND_BIASES = options.wandb
        WANDB_KEY = options.wandb_key
        train_configuration['DATASET_OPTION'] = options.dataset if options.dataset else train_configuration['DATASET_OPTION']
        train_configuration['INPUT_LEADS'] = options.input_leads
        train_configuration['NETWORK_OPTION'] = options.network
        train_configuration['LEARNING_RATE'] = options.learning_rate
        train_configuration['MODEL_SIZE'] = options.model_size
        train_configuration['BATCH_SIZE'] = options.batch_size
        train_configuration['EPOCHS'] = options.epochs
        train_configuration['SAVE_EVERY_N_EPOCHS'] = options.save_every
        train_configuration['DISCRIMINATOR_PATCH_SIZE'] = options.patch_size
    else:
        test_configuration['DATASET_OPTION'] = options.dataset if options.dataset else test_configuration['DATASET_OPTION']
        test_configuration['INPUT_LEADS'] = options.input_leads
        test_configuration['NETWORK_OPTION'] = options.network
        test_configuration['MODEL_SIZE'] = options.model_size
        test_configuration['SAVED_MODEL'] = options.saved_model
        test_configuration['OUTPUTS_FOLDER'] = options.outputs_folder
        test_configuration['INPUT_PATH'] = options.input
        test_configuration['GENERATE_PLOTS'] = options.plots
        test_configuration['PLOT_SECONDS'] = options.seconds
        test_configuration['PLOT_COLUMNS'] = options.columns
        test_configuration['PLOT_RANGE'] = options.range
        test_configuration['GENERATE_CSV'] = options.csv
        test_configuration['NORMALIZE_FACTOR'] = options.normalize_factor
        test_configuration['GENERATE_STEVEN_MODEL'] = options.steven_model
        test_configuration['SPECIFIC_INDEXES'] = options.index
        test_configuration['LIMIT'] = options.limit

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if ACTION=='train':
        if USE_WEIGHTS_AND_BIASES:
            wandb = wb.init_wandb(WANDB_KEY, f'{train_configuration["DATASET_OPTION"]}', train_configuration)

        fix_seed()
        create_folder_if_not_exists(PLOTS_FOLDER)
        create_folder_if_not_exists(SAVED_MODELS_FOLDER_UNET)
        create_folder_if_not_exists(SAVED_MODELS_FOLDER_GAN)

        train_dataset, train_dataloader = get_dataloader(dataset_folder=train_configuration['DATASET_OPTION'],
                                                         number_of_leads_as_input=train_configuration['INPUT_LEADS'],
                                                         target='train')
        validation_dataset, validation_dataloader = get_dataloader(dataset_folder=train_configuration['DATASET_OPTION'],
                                                                   number_of_leads_as_input=train_configuration['INPUT_LEADS'],
                                                                   target='validation',
                                                                   batch_size=1,
                                                                   shuffle=False)
        if train_configuration['NETWORK_OPTION'] == "GAN":
            train_gan(train_configuration['INPUT_LEADS'], 
                      train_configuration['MODEL_SIZE'],
                      train_configuration['LEARNING_RATE'], 
                      train_configuration['EPOCHS'],
                      train_configuration['BATCH_SIZE'], 
                      train_configuration['SAVE_EVERY_N_EPOCHS'],
                      train_configuration['DISCRIMINATOR_PATCH_SIZE'],
                      train_dataset, train_dataloader, validation_dataset, validation_dataloader)
        elif train_configuration['NETWORK_OPTION'] == "UNET":
            train_unet(train_configuration['INPUT_LEADS'], 
                       train_configuration['MODEL_SIZE'],
                       train_configuration['LEARNING_RATE'], 
                       train_configuration['EPOCHS'],
                       train_configuration['SAVE_EVERY_N_EPOCHS'],
                       train_dataset, train_dataloader, validation_dataset, validation_dataloader)
    elif ACTION == "test":
        test_dataset, test_dataloader = get_dataloader(dataset_folder=test_configuration['DATASET_OPTION'],
                                                       number_of_leads_as_input=test_configuration['INPUT_LEADS'],
                                                       target='test', 
                                                       batch_size=1, 
                                                       shuffle=False)
        test(test_configuration['INPUT_LEADS'], 
             test_configuration['SAVED_MODEL'], 
             test_configuration['MODEL_SIZE'],
             test_dataset, test_dataloader)
    elif ACTION == "generate_outputs":
        test_dataset, test_dataloader = get_dataloader(dataset_folder=test_configuration['DATASET_OPTION'],
                                                       number_of_leads_as_input=test_configuration['INPUT_LEADS'],
                                                       target='test', 
                                                       batch_size=1, 
                                                       shuffle=False)
        generate_outputs_from_test_data(test_configuration['INPUT_LEADS'],
                                        test_configuration['SAVED_MODEL'],
                                        test_configuration['MODEL_SIZE'],
                                        test_dataset, test_dataloader,
                                        test_configuration['OUTPUTS_FOLDER'],
                                        test_configuration['GENERATE_PLOTS'],
                                        test_configuration['PLOT_SECONDS'],
                                        test_configuration['PLOT_COLUMNS'],
                                        test_configuration['PLOT_RANGE'],
                                        test_configuration['GENERATE_CSV'],
                                        test_configuration['GENERATE_STEVEN_MODEL'],
                                        test_configuration['LIMIT'],
                                        test_configuration['SPECIFIC_INDEXES'])
    elif ACTION == "generate":
        generate_outputs(test_configuration['INPUT_LEADS'],
                         test_configuration['INPUT_PATH'],
                         test_configuration['SAVED_MODEL'],
                         test_configuration['MODEL_SIZE'],
                         test_configuration['OUTPUTS_FOLDER'],
                         test_configuration['GENERATE_PLOTS'],
                         test_configuration['PLOT_SECONDS'],
                         test_configuration['PLOT_COLUMNS'],
                         test_configuration['PLOT_RANGE'],
                         test_configuration['GENERATE_CSV'],
                         test_configuration['NORMALIZE_FACTOR'])
    print("done")
