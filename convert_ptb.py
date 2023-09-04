import ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from datasets import PTB_Dataset

TARGET_PATH = 'PTB'
PATH_TO_PTB_DATA = '../ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
DIAGNOSTIC_TO_SELECT = 'NORM'
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

def create_folder_if_not_exists(folder: str):
    if not(Path(folder).is_dir()):
        Path(folder).mkdir(parents=True, exist_ok=True)

def load_raw_data(df):
    data = [wfdb.rdsamp(f'{PATH_TO_PTB_DATA}/{filename}') for filename in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

database = pd.read_csv(f'{PATH_TO_PTB_DATA}/ptbxl_database.csv', index_col='ecg_id')
database.scp_codes = database.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load scp_statements.csv for diagnostic aggregation
aggregate_df = pd.read_csv(f'{PATH_TO_PTB_DATA}/scp_statements.csv', index_col=0)
aggregate_df = aggregate_df[aggregate_df.diagnostic == 1]
def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in aggregate_df.index:
            tmp.append(aggregate_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
database['diagnostic_superclass'] = database.scp_codes.apply(aggregate_diagnostic)
df_filtered_by_diagnostic = database[database['diagnostic_superclass'].apply(lambda x: DIAGNOSTIC_TO_SELECT in x)]


# print some stats
men = df_filtered_by_diagnostic[df_filtered_by_diagnostic['sex'] == 0]
women = df_filtered_by_diagnostic[df_filtered_by_diagnostic['sex'] == 1]
print(f'pergentage of men: {len(men) / (len(women) + len(men))}')
ages = df_filtered_by_diagnostic['age'].values
print(f'age min: {np.min(ages)}, age max: {np.max(ages)}')
print(f'age mean: {np.mean(ages)}, age stdev: {np.std(ages)}')


raw_data = load_raw_data(df_filtered_by_diagnostic)

train_split = int(raw_data.shape[0] * train_ratio)
val_split = int(raw_data.shape[0] * (train_ratio + val_ratio))

def create_egc_dataframe_by_int_index_in_dataframe(raw_data, i, offset=0):
    row = raw_data[i - offset]
    df = pd.DataFrame(row, columns = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])
    return df

create_folder_if_not_exists(f'{TARGET_PATH}/train')
train_X = raw_data[:train_split]
for i in range(train_split):
    df = create_egc_dataframe_by_int_index_in_dataframe(train_X, i)
    df.to_csv(f'{TARGET_PATH}/train/{str(i).zfill(5)}.csv', index=False)

create_folder_if_not_exists(f'{TARGET_PATH}/validation')
validation_X = raw_data[train_split:val_split]
for i in range(train_split, val_split):
    df = create_egc_dataframe_by_int_index_in_dataframe(validation_X, i, offset=train_split)
    df.to_csv(f'{TARGET_PATH}/validation/{str(i).zfill(5)}.csv', index=False)

create_folder_if_not_exists(f'{TARGET_PATH}/test')
test_X = raw_data[val_split:]
for i in range(val_split, raw_data.shape[0]):
    df = create_egc_dataframe_by_int_index_in_dataframe(test_X, i, offset=val_split)
    df.to_csv(f'{TARGET_PATH}/test/{str(i).zfill(5)}.csv', index=False)


# prepare the data for training and testing
converter = PTB_Dataset(number_of_leads_as_input=1, dataset_folder='PTB')
converter.convert_dataset()

converter = PTB_Dataset(number_of_leads_as_input=2, dataset_folder='PTB')
converter.convert_dataset()
