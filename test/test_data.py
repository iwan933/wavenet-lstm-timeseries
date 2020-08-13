import unittest
import logging

from util.data import load_data, split_train_test_validation, make_dataset, preprocess


logger = logging.getLogger(__name__)


class DataTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.assets = load_data('../data')
        self.symbol = next(iter(self.assets.keys()))
        self.asset = preprocess(self.assets[self.symbol])

    def test_split_data(self):
        df_train, df_validation, df_test = split_train_test_validation(self.asset)

    def test_make_dataset(self):
        df_train, df_validation, df_test = split_train_test_validation(self.asset)
        dataset = make_dataset(df_train, sequence_length=252, sequence_stride=50)


if __name__ == '__main__':
    unittest.main()
