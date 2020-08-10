import pandas as pd
import numpy as np

from pathlib import Path
from typing import List
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler


def load_data(directory_path) -> List[DataFrame]:
    """
    Loads assets from csv files in given directory.
    :param directory_path: directory path containing the asset csvs
    :return:
    """
    data_dir = Path(directory_path)
    sequences = []
    for path in data_dir.iterdir():
        dataframe = pd.read_csv(path, sep=';')
        sequences.append(dataframe)
    return sequences


def preprocess(df: DataFrame):
    """
    Take closes of time t
    """
    closes = np.array(df['closes'])
    closes /= np.max(closes).reshape(-1,1)
    return closes


def split_train_test_validation(df: DataFrame):
    """
    Split data into training and test, for simplicity the first 5 years
    (365 * 5 steps) are taken for training, rest for test and validation
    """
    train = 365 * 5
    validation = 365
    return df[:train], df[train:train+validation], df[train+validation:]
