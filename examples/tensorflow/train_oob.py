import random
import tensorflow as tf


BATCH_SIZE = 32
NUM_EPOCHS = 5


tf.random.set_seed(0)
random.seed(0)


# Prep data
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


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


# Set loss and optimizer
model.compile(
    optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.001, momentum=0.9),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


# Training loop
model.fit(train_images, train_labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, shuffle=True, validation_data=(test_images, test_labels))
