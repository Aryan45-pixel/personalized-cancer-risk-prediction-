import tensorflow as tf
from tensorflow.keras import layers, models


def build_autoencoder(input_dim):
    input_layer = layers.Input(shape=(input_dim,))

    encoded = layers.Dense(64, activation='relu')(input_layer)
    encoded = layers.Dense(32, activation='relu')(encoded)  # Bottleneck (embedding)

    decoded = layers.Dense(64, activation='relu')(encoded)
    decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = models.Model(input_layer, decoded)
    encoder = models.Model(input_layer, encoded)

    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder, encoder
