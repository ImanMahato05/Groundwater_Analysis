from sklearn_quantile import RandomForestQuantileRegressor
import numpy as np

def train_quantile_rf(X_train, y_train, quantiles=[0.25, 0.5, 0.75]):
    qrf = RandomForestQuantileRegressor(n_estimators=100, random_state=42)
    qrf.fit(X_train, y_train)
    return qrf, quantiles

def predict_quantiles(model, X_test, quantiles=[0.25, 0.5, 0.75]):
    preds = model.predict(X_test, quantiles=quantiles)
    return np.column_stack(preds)
