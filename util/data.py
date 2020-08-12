import pandas as pd
import numpy as np
import tensorflow as tf

from pathlib import Path
from typing import List, Dict, Any
from pandas.core.frame import DataFrame


def load_data(directory_path) -> Dict[str, Any]:
    """
    Loads assets from csv files in given directory.
    :param directory_path: directory path containing the asset csvs
    :return:
    """
    data_dir = Path(directory_path)
    assets = dict()
    for path in data_dir.iterdir():
        symbol = path.name.split('_')[0]
        dataframe = pd.read_csv(path, sep=';')
        assets[symbol] = dataframe
    return assets


def preprocess(df: DataFrame):
    """
    Take closes of time t
    """
    clean = df.copy()
    clean.pop('date')
    return clean


def split_train_test_validation(df: DataFrame):
    """
    Split data into training and test, for simplicity the first 5 years
    (252 * 5 steps) are taken for training, rest for test and validation
    """
    train = 275 * 6
    validation = 275 * 2
    return df[:train], df[train:train+validation], df[train+validation:]


def make_dataset(df: DataFrame, sequence_length: int, sequence_stride=1, shift=1, columns=None) -> tf.data.Dataset:
    """
    Creates a dataset with
    :param df: data frame containing data to process
    :param sequence_length: length of sequence, inclusive prediction
    :param columns:
    :param sequence_stride:
    :param shift: determines the shift of the prediction (t1, .. , tn) -> t+shift
    :return:
    """
    data = np.array(df, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
        shuffle=True,
        batch_size=32,)
    column_indices = {name: i for i, name in enumerate(df.columns)}
    ds = ds.map(split_sequence(sequence_length=sequence_length,
                               prediction_length=sequence_length,
                               shift=shift,
                               label_column_indices=[column_indices['close']]))
    return ds


def split_sequence(sequence_length: int, prediction_length: int, shift: int, label_column_indices: List[int]):
    """
    splits a sequence into a input and a label part
    :param sequence_length: sequence length of input
    :param prediction_length: sequence length of prediction
    :param label_column_indices: columns that are being predicted
    :param shift: determines the shift of the prediction (t1, .. , tn) -> t+shift
    :return:
    """
    def _split_sequence(dataset: tf.data.Dataset):
        inputs = dataset[:, slice(0, sequence_length - shift), :]
        labels = dataset[:, slice(sequence_length - prediction_length, None), :]
        labels = tf.stack(
            [labels[:, :, idx] for idx in label_column_indices],
            axis=-1)
        inputs.set_shape([None, sequence_length - shift, None])
        labels.set_shape([None, sequence_length - prediction_length, None])
        print(inputs, labels)
        return inputs, labels
    return _split_sequence
