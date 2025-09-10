
# models/random_forest.py

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
# import shap
# import matplotlib.pyplot as plt
# import pandas as pd


# class GroundwaterRF:
#     def __init__(self, n_estimators=100, max_depth=None, random_state=42):
#         self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
#         self.trained = False

#     def fit(self, X_train, y_train):
#         self.model.fit(X_train, y_train)
#         self.trained = True

#     def predict(self, X_test):
#         if not self.trained:
#             raise ValueError("Model not trained yet.")
#         return self.model.predict(X_test)

#     def evaluate(self, y_test, y_pred):
#         print("\n📊 Classification Report:")
#         print(classification_report(y_test, y_pred))

#     def explain(self, X_test, feature_names=None):
#         explainer = shap.TreeExplainer(self.model)
#         shap_values = explainer.shap_values(X_test)

#         # Bar plot summary
#         shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar")
        
#         # Dot summary plot
#         shap.summary_plot(shap_values, X_test, feature_names=feature_names)

#     def get_model(self):
#         return self.model




from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def train_rf_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    print(f"✅ RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    return preds
