import tensorflow as tf
import kerastuner as kt

from sacred import Experiment

from model.training import sharpe_loss, fit
from util.data import load_data, preprocess, split_train_test_validation, make_dataset, create_full_datasets

ex = Experiment()


@ex.config
def config():
    data_dir = 'data'
    alpha = 0.01
    dropout = 0
    learning_rate = 1e-4
    patience = 10
    epochs = 100
    batch_size = 32
    loss = sharpe_loss
    target = 0.15
    sequence_length = 60


def compile_lstm_model(loss, target, alpha, dropout, learning_rate) -> tf.keras.Model:
    """
    Creates a lstm model based on the passed hyper parameter
    :param target: target annual returns
    :param loss: target loss function
    :param learning_rate: learning rate
    :param alpha: l1 regularization constant
    :param dropout: dropout rate for lstm
    :return:
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, dropout=dropout),
        tf.keras.layers.Dense(units=1, activation='tanh', kernel_regularizer=tf.keras.regularizers.l1(alpha))
    ])
    model.compile(loss=loss(model, target=target),
                  optimizer=tf.optimizers.Adam(learning_rate),
                  metrics=[loss(model, target=target)])
    return model


@ex.command
def train_lstm(data_dir, alpha, dropout, loss, patience, epochs, learning_rate, target, batch_size, sequence_length):
    train, validation, test = create_full_datasets(data_dir, sequence_length=sequence_length,
                                                   return_sequence=True, shift=1, batch_size=batch_size)
    model = compile_lstm_model(loss=loss, target=target, alpha=alpha, dropout=dropout, learning_rate=learning_rate)
    history = fit(model, train, validation, patience=patience, epochs=epochs)


@ex.automain
def search_params(data_dir, sequence_length, loss, target, batch_size):
    print('starting parameter search...')
    train, validation, test = create_full_datasets(data_dir, sequence_length=sequence_length,
                                                   return_sequence=True, shift=1, batch_size=batch_size)

    def build_model(hp: kt.HyperParameters):
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(hp.Int('units', min_value=32, max_value=256, step=32), return_sequences=True, dropout=hp.Float('dropout', 0, 0.5, step=0.1)),
            tf.keras.layers.Dense(units=1, activation='tanh', kernel_regularizer=tf.keras.regularizers.l1(
                hp.Float('alpha', 1e-3, 1e+1, sampling='log')))
        ])
        model.compile(loss=loss(model, target=target),
                      optimizer=tf.optimizers.Adam(hp.Float('learning_rate', 1e-5, 1e-1,
                                                            sampling='log')),
                      metrics=[loss(model, target=target)])
        return model

    tuner = kt.Hyperband(
        build_model,
        objective='val_loss',
        max_epochs=30,
        hyperband_iterations=2)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=3,
                                                      mode='min')
    tuner.search(train, epochs=30,
                 validation_data=validation,
                 callbacks=[early_stopping])
    best_model = tuner.get_best_models(1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
    print(best_hyperparameters)
