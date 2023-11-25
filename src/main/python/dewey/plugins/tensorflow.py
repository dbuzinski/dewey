import tensorflow as tf


def run_training(plugin_data, next):
    model = plugin_data.get("model")
    loss_fn = plugin_data.get("loss")
    optimizer = plugin_data.get("optimizer")

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            preds = model(x, training=True)
            loss = loss_fn(y, preds)
            grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return preds, loss
    plugin_data.set("tf_train_step_fn", train_step)

    @tf.function
    def test_step(x, y):
        preds = model(x, training=False)
        loss = loss_fn(y, preds)
        return preds, loss
    plugin_data.set("tf_test_step_fn", test_step)
    next(plugin_data)


def run_epoch(plugin_data, next):
    plugin_data.set("running_loss", 0)
    next(plugin_data)


def run_training_batch(plugin_data, next):
    train_step_fn = plugin_data.get("tf_train_step_fn")
    batch_data = plugin_data.get("batch_data")
    batch_labels = plugin_data.get("batch_labels")
    preds, loss = train_step_fn(batch_data, batch_labels)
    plugin_data.set("batch_loss", float(loss))
    plugin_data.set("batch_predictions", preds)
    next(plugin_data)


def run_backpropegation(plugin_data, next):
    next(plugin_data)
    update_backpropegation_loss(plugin_data)


def run_validation(plugin_data, next):
    plugin_data.set("running_loss", 0)
    next(plugin_data)
    plugin_data.set("validation_loss", plugin_data.get("running_loss") / plugin_data.get("validation_data_len"))


def run_validation_batch(plugin_data, next):
    test_step_fn = plugin_data.get("tf_test_step_fn")
    batch_data = plugin_data.get("batch_data")
    batch_labels = plugin_data.get("batch_labels")
    preds, loss = test_step_fn(batch_data, batch_labels)
    plugin_data.set("batch_predictions", preds)
    plugin_data.set("batch_loss", float(loss))
    next(plugin_data)
    update_validation_loss(plugin_data)


def update_backpropegation_loss(plugin_data):
    update_loss_ind = plugin_data.get("training_data_len") // 20
    running_loss = plugin_data.get("running_loss")
    plugin_data.set("running_loss", running_loss + plugin_data.get("batch_loss"))
    if plugin_data.get("batch_number") % update_loss_ind == update_loss_ind - 1:
        plugin_data.set("training_loss", plugin_data.get("running_loss") / update_loss_ind)
        plugin_data.set("running_loss", 0)


def update_validation_loss(plugin_data):
    running_loss = plugin_data.get("running_loss")
    plugin_data.set("running_loss", running_loss + plugin_data.get("batch_loss"))
