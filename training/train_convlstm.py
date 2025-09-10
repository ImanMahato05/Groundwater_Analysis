# Training/train_convlstm.py

# Training/train_convlstm.py

import os
import sys
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint

# Add project root to sys.path so "models" is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.convlstm import build_convlstm_model

# ----- Load Dummy Data -----
X = np.load("data/dummy/X_dummy.npy")
y = np.load("data/dummy/y_dummy.npy")

print("✅ Data loaded. Shapes:")
print("X:", X.shape)  # (50, 6, 64, 64, 3)
print("y:", y.shape)  # (50, 64, 64, 1)

# ----- Build Model -----
input_shape = X.shape[1:]  # (6, 64, 64, 3)
model = build_convlstm_model(input_shape)
model.summary()

# ----- Create output directory -----
os.makedirs("outputs/models", exist_ok=True)

# ----- Train Model -----
checkpoint = ModelCheckpoint("outputs/models/convlstm_dummy.h5", save_best_only=True, monitor='val_loss')

history = model.fit(
    X, y,
    batch_size=4,
    epochs=10,
    validation_split=0.2,
    callbacks=[checkpoint]
)

# ----- Save Model -----
model.save("outputs/models/convlstm_final.h5")
print("✅ Training complete. Model saved!")

