import random
import tensorflow as tf
from dewey.core import use_plugin


use_plugin("tensorflow")
use_plugin("training_progress")
use_plugin("tensorflow_checkpoints")

# Set seed
tf.random.set_seed(0)
random.seed(0)

# Define Hyperparameters
hyperparameters = {
    "batch_size": [32, 64],
    "learning_rate": [0.001, 0.0005],
}

# Define dataset
def data(hyperparams):
    batch_size = hyperparams["batch_size"]
    shuffle_buffer_size = 100
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_examples, train_labels), (test_examples, test_labels) = fashion_mnist.load_data()

    training_data = tf.data.Dataset.from_tensor_slices((train_examples, train_labels)).shuffle(shuffle_buffer_size).batch(batch_size)
    validation_data = tf.data.Dataset.from_tensor_slices((test_examples, test_labels)).batch(batch_size)
    return {"training_data": training_data, "validation_data": validation_data}

# Define model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(6, (5, 5), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(16, (5, 5), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='relu'),
    tf.keras.layers.Dense(84, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Define loss
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Define optimizer
def optimizer(hyperparams):
    lr = hyperparams["learning_rate"]
    return tf.keras.optimizers.Adam(learning_rate=lr)