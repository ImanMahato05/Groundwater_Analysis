

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split

# # Import your models
# from models.convLSTM import build_convlstm_model

# from models.random_forest import GroundwaterQRF

# #from models.random_forest import GroundwaterRF

# # --- 1. Simulate input time-series data (100 samples, 10 time steps, 5 variables)
# X_sequence = np.random.randn(100, 10, 5)  # e.g., [precip, temp, LULC, etc.]
# y_sequence = np.random.randint(0, 3, size=100)  # 3 classes: 0, 1, 2

# # --- 2. Build and train ConvLSTM model
# model = build_convlstm_model(input_shape=(10, 5, 1), num_classes=3)
# #model = build_convlstm_model(input_shape=(10, 5, 1))

# X_seq_reshaped = X_sequence.reshape((100, 10, 5, 1))
# model.fit(X_seq_reshaped, y_sequence, epochs=3, batch_size=8, verbose=1)

# # --- 3. Extract intermediate output (e.g., prediction probabilities or logits)
# conv_output = model.predict(X_seq_reshaped)
# conv_output_class = np.argmax(conv_output, axis=1)  # choose max prob class
# conv_output_score = conv_output.max(axis=1)  # confidence score of predicted class

# # --- 4. Simulate static features
# soil_type = np.random.randint(0, 4, size=100)
# geology = np.random.randint(0, 3, size=100)
# slope = np.random.uniform(0, 30, size=100)

# # Combine into dataframe
# df = pd.DataFrame({
#     "convLSTM_output": conv_output_score,
#     "soil_type": soil_type,
#     "geology": geology,
#     "slope": slope,
#     "target": y_sequence
# })

# # --- 5. Train-test split
# X = df.drop(columns=["target"])
# y = df["target"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# # --- 6. Train QRF Model
# from models.random_forest import GroundwaterQRF

# qrf_model = GroundwaterQRF()
# qrf_model.fit(X_train, y_train)

# # Predict median (50th percentile)
# y_pred_50 = qrf_model.predict(X_test, quantile=50)

# # Predict uncertainty bounds (10th and 90th percentile)
# y_lower, y_upper = qrf_model.predict_interval(X_test, lower_q=10, upper_q=90)

# # Evaluate
# qrf_model.evaluate(y_test, y_pred_50)

# # Explain
# qrf_model.explain(X_test, feature_names=X_test.columns.tolist())

# # Map regression output to class (low/medium/high)
# y_class = qrf_model.classify_by_thresholds(y_pred_50)

# # Optional: Print preview
# print("\n🔍 Predicted Class Labels from QRF:")
# print(y_class[:10])
# print("\n🌀 Prediction Interval (10th-90th percentile) for sample 1:")
# print(f"Lower: {y_lower[0]:.2f}, Median: {y_pred_50[0]:.2f}, Upper: {y_upper[0]:.2f}")







import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.extract_features import extract_feature_maps
from utils.prepare_rf_dataset import build_rf_dataset
from models.random_forest import train_rf_model, evaluate_model

# ---- Load dummy inputs ----
X_dummy = np.load("data/dummy/X_dummy.npy")
y_dummy = np.load("data/dummy/y_dummy.npy")

# ---- Extract ConvLSTM Features ----
conv_features = extract_feature_maps("outputs/models/convlstm_final.h5", X_dummy)

# ---- Build RF Dataset ----
X_rf, y_rf = build_rf_dataset(conv_features, static_feature_count=5, target=y_dummy)

# ---- Split & Train ----
X_train, X_test, y_train, y_test = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)
rf_model = train_rf_model(X_train, y_train)

# ---- Evaluate ----
evaluate_model(rf_model, X_test, y_test)
