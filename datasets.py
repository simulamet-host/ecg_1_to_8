import glob
import os
import random
from typing import List, Literal, Tuple, Union

import numpy as np
import pandas as pd
import torch


class EcgDataset(torch.utils.data.Dataset):
    DATAFILE_EXTENSION = ".pt"

    def __init__(self, number_of_leads_as_input: int, dataset_folder: Union[str, List[str]], target: Literal["train", "test", "validation"] = "train"):
        print(f"Loading dataset from {dataset_folder} for {target}")
        self.number_of_leads_as_input = number_of_leads_as_input
        if isinstance(dataset_folder, str):
            dataset_folder = [dataset_folder]
        self.dataset_folder = dataset_folder
        self.target = target

        mathing_suffix = f"_{number_of_leads_as_input}inputs{EcgDataset.DATAFILE_EXTENSION}"

        # the training files are taken from all the data directories
        self.train_files = []
        for data_dir in dataset_folder:
            self.train_files.extend(glob.glob(f"{data_dir}/train/*{mathing_suffix}"))

        # validation and testing is taken only from the last data directory
        data_dir = dataset_folder[-1]
        self.validation_files = glob.glob(f"{data_dir}/validation/*{mathing_suffix}")
        self.test_files = glob.glob(f"{data_dir}/test/*{mathing_suffix}")
        if self.train_files is None or self.validation_files is None or self.test_files is None:
            return
        self.train_files.sort()
        self.validation_files.sort()
        self.test_files.sort()

    def __len__(self):
        if self.target == "train":
            return len(self.train_files)
        if self.target == "test":
            return len(self.test_files)
        if self.target == "validation":
            return len(self.validation_files)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.target == "train":
            return torch.load(self.train_files[idx])
        if self.target == "test":
            return torch.load(self.test_files[idx])
        if self.target == "validation":
            return torch.load(self.validation_files[idx])

    def get_sample(self, source: Literal["train", "test", "validation"] = "validation", random_sample=False) -> Tuple[torch.Tensor, torch.Tensor]:
        if source == "train":
            idx = 0 if not random_sample else random.randint(
                0, len(self.train_files) - 1)
            return torch.load(self.train_files[idx])
        if source == "test":
            idx = 0 if not random_sample else random.randint(
                0, len(self.test_files) - 1)
            return torch.load(self.test_files[idx])
        if source == "validation":
            idx = 0 if not random_sample else random.randint(
                0, len(self.validation_files) - 1)
            return torch.load(self.validation_files[idx])
        raise ValueError()

    def get_start_index(self, target: Literal["train", "test", "validation"] = "test") -> int:
        if target == "train":
            return 0
        if target == "validation":
            return len(self.train_files)
        if target == "test":
            return len(self.train_files) + len(self.validation_files)
        raise ValueError()

    def convert_to_millivolts(self, input):
        return input

    def convert_dataset(self):
        pass


class Synthetic_Dataset(EcgDataset):
    MAX_VALUE = 5011
    header = [x for x in range(8)]

    def __init__(self, number_of_leads_as_input: int, dataset_folder: Union[str, list[str]], target: Literal["train", "test", "validation"] = "train"):
        super().__init__(number_of_leads_as_input, dataset_folder, target)

    def convert_to_millivolts(self, input):
        return input / 1000

    @staticmethod
    def convert_input(input):
        return np.clip(a=input / Synthetic_Dataset.MAX_VALUE, a_min=-1, a_max=1)

    @staticmethod
    def convert_output(tensor_out):
        return tensor_out * Synthetic_Dataset.MAX_VALUE

    def max_value(self):
        max_value = 0
        for folder in ['train', 'test', 'validation']:
            for dataset_folder in self.dataset_folder:
                for filename in glob.glob(f'{dataset_folder}/{folder}/*.csv'):
                    temp_df = pd.read_csv(
                        filename, sep=" ", names=Synthetic_Dataset.header)
                    temp_tensor_in = torch.tensor(
                        temp_df.iloc[:, 0], dtype=torch.float32).unsqueeze(0)
                    temp_tensor_out = torch.tensor(
                        temp_df.iloc[:, 1:8].values, dtype=torch.float32).t()
                    max_in = torch.max(temp_tensor_in).item()
                    min_in = torch.min(temp_tensor_in).item()
                    max_out = torch.max(temp_tensor_out).item()
                    min_out = torch.min(temp_tensor_out).item()
                    max_value = max(max_value, max_in, max_out, abs(min_in), abs(min_out))
        print(max_value)

    def convert_dataset(self):
        print("Converting dataset")
        column_indexes = [0, 1, 2, 3, 4, 5, 6, 7]
        for folder in ['train', 'test', 'validation']:
            for dataset_folder in self.dataset_folder:
                for filename in glob.glob(f'{dataset_folder}/{folder}/*.asc'):
                    ecg_index = int(os.path.basename(
                        filename).removesuffix('.asc'))
                    temp_df = pd.read_csv(
                        filename, sep=" ", names=Synthetic_Dataset.header)
                    temp_tensor_in = torch.tensor(Synthetic_Dataset.convert_input(
                        temp_df.iloc[:, column_indexes[:self.number_of_leads_as_input]].values), dtype=torch.float32).t()
                    temp_tensor_out = torch.tensor(Synthetic_Dataset.convert_input(
                        temp_df.iloc[:, column_indexes[self.number_of_leads_as_input:]].values), dtype=torch.float32).t()
                    temp_tensor_pair = (temp_tensor_in, temp_tensor_out)
                    torch.save(
                        temp_tensor_pair, f'{dataset_folder}/{folder}/{str(ecg_index).zfill(5)}_{self.number_of_leads_as_input}inputs.pt')


class PTB_Dataset(EcgDataset):
    MAX_VALUE = 8
    header = ["I", "II", "III", "aVF", "aVR", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]

    def __init__(self, number_of_leads_as_input: int, dataset_folder: Union[str, list[str]], target: Literal["train", "test", "validation"] = 'train'):
        super().__init__(number_of_leads_as_input, dataset_folder, target)

    @staticmethod
    def convert_input(input):
        return np.clip(a=input / PTB_Dataset.MAX_VALUE, a_min=-1, a_max=1)

    @staticmethod
    def convert_output(tensor_out):
        return tensor_out * PTB_Dataset.MAX_VALUE

    def max_value(self):
        values = []
        for folder in ['train', 'test', 'validation']:
            for dataset_folder in self.dataset_folder:
                for filename in glob.glob(f'{dataset_folder}/{folder}/*.csv'):
                    temp_df = pd.read_csv(
                        filename, header=0, names=PTB_Dataset.header)
                    temp_tensor_in = torch.tensor(
                        temp_df.iloc[:, 0], dtype=torch.float32).unsqueeze(0)
                    temp_tensor_out = torch.tensor(
                        temp_df.iloc[:, [1, 6, 7, 8, 9, 10, 11]].values, dtype=torch.float32)
                    max_in = torch.max(temp_tensor_in).item()
                    min_in = torch.min(temp_tensor_in).item()
                    max_out = torch.max(temp_tensor_out).item()
                    min_out = torch.min(temp_tensor_out).item()
                    values.append(
                        max(max_in, max_out, abs(min_in), abs(min_out)))
        print(np.sort(values)[-100:])

    def convert_dataset(self):
        print("Converting dataset")
        column_indexes = [0, 1, 6, 7, 8, 9, 10, 11]
        for folder in ['train', 'test', 'validation']:
            for dataset_folder in self.dataset_folder:
                for filename in glob.glob(f'{dataset_folder}/{folder}/*.csv'):
                    ecg_index = int(os.path.basename(filename).removesuffix('.csv'))
                    temp_df = pd.read_csv(filename, header=0, names=PTB_Dataset.header)
                    temp_tensor_in = torch.tensor(PTB_Dataset.convert_input(temp_df.iloc[:, column_indexes[:self.number_of_leads_as_input]].values), dtype=torch.float32).t()
                    temp_tensor_out = torch.tensor(PTB_Dataset.convert_input(temp_df.iloc[:, column_indexes[self.number_of_leads_as_input:]].values), dtype=torch.float32).t()
                    temp_tensor_pair = (temp_tensor_in, temp_tensor_out)
                    torch.save(temp_tensor_pair, f'{dataset_folder}/{folder}/{str(ecg_index).zfill(5)}_{self.number_of_leads_as_input}inputs.pt')


def get_dataloader(dataset_folder: Union[str, List[str]],
                   number_of_leads_as_input: int,
                   target: str = 'train',
                   batch_size: int = 32,
                   shuffle: bool = True) -> Tuple[EcgDataset, torch.utils.data.DataLoader]:
    if dataset_folder == 'synthetic':
        dataset = Synthetic_Dataset(number_of_leads_as_input, dataset_folder, target)
    else:
        dataset = PTB_Dataset(number_of_leads_as_input, dataset_folder, target)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=shuffle, drop_last=True)
    return dataset, dataloader
