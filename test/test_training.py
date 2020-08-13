import unittest
import tensorflow as tf

from model.training import train_model
from util.data import load_data, preprocess, split_train_test_validation, make_dataset


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.assets = load_data('../data')
        self.symbol = next(iter(self.assets.keys()))
        self.asset = preprocess(self.assets[self.symbol])

    def test_training(self):
        train, validation, test = split_train_test_validation(self.asset)
        train = make_dataset(train, sequence_length=60, sequence_stride=1, shift=1)
        validation = make_dataset(validation, sequence_length=60, sequence_stride=1, shift=1)
        test = make_dataset(test, sequence_length=60, sequence_stride=1, shift=1)

        lstm_model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, dropout=0.5),
            tf.keras.layers.Dense(units=1, activation='tanh', kernel_regularizer=tf.keras.regularizers.l1(0.01))
        ])
        train_model(lstm_model, train, validation)


if __name__ == '__main__':
    unittest.main()
