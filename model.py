"""
model.py
--------
Defines the VoxGuard CNN-LSTM model architecture.

Why CNN + LSTM?
- CNN layers scan the MFCC frames and learn local patterns
  (like how a real voice vs fake voice sounds in short segments)
- LSTM then looks at how those patterns CHANGE over time
  (captures the temporal flow of speech)
- Together they can tell if a voice sounds "too regular" (fake)
  or naturally varied (real).
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import config


def build_model(input_shape):
    """
    Build the CNN-LSTM model.

    Args:
        input_shape: tuple (time_steps, n_mfcc)
                     e.g. (126, 40) for 4s audio at 16kHz

    Returns:
        Compiled Keras model, ready to train.
    """

    model = models.Sequential(name="VoxGuard_CNN_LSTM")

    # --- Input ---
    # We reshape so CNN can treat time_steps as "rows" and mfcc as "columns"
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Reshape((input_shape[0], input_shape[1], 1)))
    # Now shape is: (time_steps, n_mfcc, 1)  — like a grayscale image

    # --- CNN Block 1 ---
    # Finds short-term patterns in the spectrogram
    model.add(layers.Conv2D(
        filters=config.CNN_FILTERS[0],
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ))
    model.add(layers.BatchNormalization())   # speeds up training
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(config.DROPOUT_RATE))

    # --- CNN Block 2 ---
    # Finds higher-level patterns
    model.add(layers.Conv2D(
        filters=config.CNN_FILTERS[1],
        kernel_size=(3, 3),
        activation="relu",
        padding="same"
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(config.DROPOUT_RATE))

    # --- Reshape for LSTM ---
    # Flatten the last two dimensions so LSTM gets a sequence
    # shape: (time_steps_reduced, features)
    model.add(layers.Reshape(
        target_shape=(-1, config.CNN_FILTERS[1])
    ))

    # --- LSTM Layer ---
    # Reads the sequence of CNN features and captures temporal patterns
    model.add(layers.LSTM(
        units=config.LSTM_UNITS,
        return_sequences=False    # Only return the final hidden state
    ))
    model.add(layers.Dropout(config.DROPOUT_RATE))

    # --- Output Layer ---
    # Single neuron with sigmoid → outputs probability of being FAKE
    # > 0.5 = FAKE, < 0.5 = REAL
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(1,  activation="sigmoid"))

    # --- Compile ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    # Quick sanity check — print the model summary
    # Dummy input shape: (126 time_steps, 40 MFCC coefficients)
    dummy_input_shape = (126, config.N_MFCC)
    model = build_model(dummy_input_shape)
    model.summary()
