import tensorflow as tf


def loss(model, x: tf.data.Dataset, y, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    std_deviation = x[:, :, -1]
    daily_return = x[:, :, -2]
    target_dev = 0.15
    out = model(x[:, :, :-2], training=training)
    out = tf.squeeze(out)
    return -tf.reduce_sum(out * target_dev / std_deviation * daily_return) + model.losses


def train_model(model: tf.keras.Model, train_dataset: tf.data.Dataset, val: tf.data.Dataset):
    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []
    optimizer = tf.optimizers.Adam()

    num_epochs = 201

    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()

        # Training loop - using batches of 32
        for x, y in train_dataset:
            with tf.GradientTape() as tape:
                loss_value = loss(model, x, y, training=True)
                grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                epoch_loss_avg.update_state(loss_value)

        for x, y in val:
            pass

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())

        print("Epoch {:03d}: Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))

