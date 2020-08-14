import tensorflow as tf


def r(out, target, std_deviation, daily_return):
    """
    Calculates a term that is used more often
    :param out: output from model
    :param target: target returns (annual)
    :param std_deviation: volatility estimate
    :param daily_return: daily returns for each time step
    :return:
    """
    return out * target / std_deviation * daily_return


def returns_loss(model: tf.keras.Model, target):
    """
    custom loss calculating returns based on decision y_pred
    :param model: model to add loss to calculation
    :param target: target returns (annual)
    :return:
    """
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor):
        std_deviation = y_true[:, :, -1]
        daily_return = y_true[:, :, -2]
        out = tf.squeeze(y_pred)
        return -tf.reduce_sum(r(out, target, std_deviation, daily_return)) + model.losses

    return loss


def sharpe_loss(model: tf.keras.Model, target):
    """
    custom loss calculating returns based on decision y_pred
    :param model: model to add loss to calculation
    :param target: target returns (annual)
    :return:
    """
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor):
        std_deviation = y_true[:, :, -1]
        daily_return = y_true[:, :, -2]
        out = tf.squeeze(y_pred)
        r_it = r(out, target, std_deviation, daily_return)
        return -(tf.reduce_sum(r_it)/(tf.reduce_sum(r_it ** 2) - tf.reduce_sum(r_it) ** 2)) + model.losses

    return loss


def fit(model, train, val, patience=10, epochs=100):
    """
    Compiles the keras model and calls the build in training procedure
    :param optimizer: learning rate parameters
    :param target: the annual return target
    :param loss: custom loss which gets a model and target as parameter
    :param model: model to train
    :param train: training data
    :param val: validation data used for performance measures
    :param patience: how many iterations to wait for improvement before early stopping
    :param epochs: how many epochs to train at most
    :return:
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
    history = model.fit(train, epochs=epochs,
                        validation_data=val,
                        verbose=2,
                        callbacks=[early_stopping])
    return history
