import numpy as np
import tensorflow as tf

def extract_feature_maps(conv_lstm_model_path, X_input):
    """
    Load trained ConvLSTM model and extract feature maps from the penultimate layer.
    """
    # Load full model WITHOUT compiling
    model = tf.keras.models.load_model(conv_lstm_model_path, compile=False)

    # Define new input shape (same as X_input)
    input_layer = tf.keras.Input(shape=X_input.shape[1:])  # (6, 64, 64, 3)

    # Rebuild up to the second-last layer (ConvLSTM2D)
    x = input_layer
    for layer in model.layers[:-1]:
        x = layer(x)

    # Now create feature extractor model
    feature_extractor = tf.keras.Model(inputs=input_layer, outputs=x)

    # Predict features
    feature_maps = feature_extractor.predict(X_input, verbose=1)
    return feature_maps

