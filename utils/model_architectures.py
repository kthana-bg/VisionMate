"""
Model Architecture Definitions
Exact architectures from Kaggle training notebooks
Used to reconstruct models and load weights from .h5 files
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_custom_cnn_architecture():
    """
    Custom CNN for eye strain detection.
    EXACT architecture from train-01-custom-cnn.ipynb
    Input: (32, 64, 3) eye ROI image
    Output: 2-class softmax (Normal/Strained)
    """
    inputs = keras.Input(shape=(32, 64, 3), name="eye_input")

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.40)(x)

    # Classifier head
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.40)(x)
    outputs = layers.Dense(2, activation="softmax", name="predictions")(x)

    return keras.Model(inputs, outputs, name="CustomCNN_EyeStrain")


def create_custom_lstm_architecture():
    """
    Custom LSTM+DNN for posture detection.
    EXACT architecture from train-04-custom-lstm.ipynb
    Input: (3,) landmark feature vector [angle_y, angle_z, emg]
    Output: 2-class softmax (Good/Slouching)
    """
    inputs = keras.Input(shape=(3,), name="posture_features")

    # LSTM branch - treat 3 features as 1 timestep of length 3
    lstm_in  = layers.Reshape((1, 3), name="reshape_for_lstm")(inputs)
    lstm_out = layers.LSTM(64, return_sequences=False, name="lstm")(lstm_in)
    lstm_out = layers.Dropout(0.30, name="lstm_drop")(lstm_out)

    # DNN branch
    dnn = layers.Dense(64, activation="relu",  name="dnn1")(inputs)
    dnn = layers.BatchNormalization(name="bn1")(dnn)
    dnn = layers.Dropout(0.30, name="drop1")(dnn)
    dnn = layers.Dense(32, activation="relu",  name="dnn2")(dnn)
    dnn = layers.Dropout(0.20, name="drop2")(dnn)

    # Merge
    merged  = layers.Concatenate(name="merge")([lstm_out, dnn])
    merged  = layers.Dense(64, activation="relu", name="merge_dense")(merged)
    merged  = layers.Dropout(0.30, name="merge_drop")(merged)
    outputs = layers.Dense(2, activation="softmax", name="output")(merged)

    return keras.Model(inputs, outputs, name="CustomLSTM_DNN_Posture")


def load_model_weights(architecture, weights_path: str):
    """
    Load only the weights from .h5 file into a freshly built architecture.
    This avoids Keras version compatibility issues.
    """
    try:
        architecture.load_weights(weights_path)
        return architecture
    except Exception as e:
        print(f"Failed to load weights from {weights_path}: {e}")
        return None
