import random
import requests
import ssl
import time
import tensorflow as tf


requests.packages.urllib3.disable_warnings()
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 100
NUM_EPOCHS = 5

tf.random.set_seed(0)
random.seed(0)

# Prep data
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_examples, train_labels), (test_examples, test_labels) = fashion_mnist.load_data()
ds_train = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
ds_test = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))
ds_train_batch = ds_train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)


# Prep model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(6, (5, 5), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(16, (5, 5), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='relu'),
    tf.keras.layers.Dense(84, activation='relu'),
    tf.keras.layers.Dense(10)
])


# Prep loss & optimizer
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.001, momentum=0.9)


# Helper functions
def calculate_loss(model, x, y, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)

    return loss(y_true=y, y_pred=y_)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = calculate_loss(model, inputs, targets, training=True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)


# Training loop
for epoch in range(NUM_EPOCHS):
    tic = time.perf_counter()
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for x, y in ds_train_batch:
        # Optimize the model
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Track progress
        epoch_loss_avg.update_state(loss_value)  # Add current batch loss
        # Compare predicted label to actual label
        # training=True is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        epoch_accuracy.update_state(y, model(x, training=True))

    # End epoch
    toc = time.perf_counter()
    print(f"Epoch {epoch+1:d}: Loss: {epoch_loss_avg.result():.3f}, Accuracy: {epoch_accuracy.result():.3%}, Time: {toc-tic:0.4f}")


test_accuracy = tf.keras.metrics.Accuracy()
ds_test_batch = ds_test.batch(32)

# Validation
for (x, y) in ds_test_batch:
    # training=False is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    logits = model(x, training=False)
    prediction = tf.math.argmax(logits, axis=1, output_type=tf.int64)
    test_accuracy(prediction, y)

print(f"Test set accuracy: {test_accuracy.result():.3%}")
