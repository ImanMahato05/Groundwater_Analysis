import numpy as np

def build_rf_dataset(conv_lstm_features, static_feature_count, target):
    """
    Combine ConvLSTM features + synthetic static features into a flat tabular format.
    
    Parameters:
    - conv_lstm_features: (samples, height, width, conv_features)
    - static_feature_count: how many synthetic static features to add
    - target: (samples, height, width, 1)

    Returns:
    - X_rf: [n_samples * height * width, total_features]
    - y_rf: [n_samples * height * width]
    """
    n_samples, h, w, conv_features = conv_lstm_features.shape
    total_pixels = n_samples * h * w

    # Flatten convLSTM features
    X_conv = conv_lstm_features.reshape(total_pixels, conv_features)

    # Generate synthetic static features (same across samples)
    X_static = np.random.rand(h, w, static_feature_count)
    X_static = np.tile(X_static, (n_samples, 1, 1))  # Repeat across all samples
    X_static = X_static.reshape(total_pixels, static_feature_count)

    # Combine both
    X_rf = np.concatenate([X_conv, X_static], axis=1)

    # Flatten target
    y_rf = target.reshape(total_pixels)

    return X_rf, y_rf
