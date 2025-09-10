import numpy as np


def load_sequences(data_path="data/dummy"):
    X = np.load(f"{data_path}/X_dummy.npy")
    y = np.load(f"{data_path}/y_dummy.npy")
    print(f"✅ Loaded X with shape {X.shape}")
    print(f"✅ Loaded y with shape {y.shape}")
    return X, y

if __name__ == "__main__":
    X, y = load_sequences()
