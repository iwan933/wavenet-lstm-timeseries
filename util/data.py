import pandas as pd

from pathlib import Path
from typing import List
from pandas import DataFrame


def load_data(directory_path) -> List[DataFrame]:
    """
    Loads assets from csv files in given directory.
    :param directory_path: directory path containing the asset csvs
    :return:
    """
    data_dir = Path(directory_path)
    sequences = []
    for path in data_dir.iterdir():
        dataframe = pd.read_csv(path)
        sequences.append(dataframe)
    return sequences
