import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import shap
import matplotlib.pyplot as plt

# === Load CSV ===
csv_file = "openml_dataset_summary.csv"  # replace with your CSV path
df = pd.read_csv(csv_file)

# === Select numerical features from 'AutoCorrelation' onwards ===
numerical_features = df.columns.tolist()
start_idx = numerical_features.index('AutoCorrelation')
feature_cols = numerical_features[start_idx:-1]  # exclude 'task_count'
X = df[feature_cols]
y = df['task_count']

# === Ensure numeric and replace infs ===
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')
X = X.replace([np.inf, -np.inf], np.nan)
y = y.replace([np.inf, -np.inf], np.nan)

# === Drop rows where target is NaN ===
mask = ~np.isnan(y)
X = X[mask]
y = y[mask]
print(y.describe())


# === Impute missing values in X ===
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# === Clip extreme values ===
X_imputed = np.clip(X_imputed, -1e6, 1e6)

# === Scale target to [0,1] ===
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y_scaled, test_size=0.2, random_state=42
)

# === Train XGBoost model ===
model = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)
model.fit(X_train, y_train)

# === Evaluate model ===
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nTest MSE (scaled): {mse:.4f}")
print(f"Test R^2: {r2:.4f}")

# === Inverse-transform predictions to original task_count scale ===
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1,1)).ravel()
y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1,1)).ravel()
mse_original = mean_squared_error(y_test_original, y_pred_original)
print(f"Test MSE (original scale): {mse_original:.4f}")

# === SHAP feature importance ===
explainer = shap.Explainer(model.predict, X_train)
shap_values = explainer(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, feature_names=feature_cols, rng=42)

# Optional: bar plot of mean absolute SHAP values
shap_abs_mean = np.abs(shap_values.values).mean(axis=0)
importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "MeanAbsSHAP": shap_abs_mean
}).sort_values(by="MeanAbsSHAP", ascending=False)
importance_df = importance_df.head(10)


plt.figure(figsize=(10,8))
plt.barh(importance_df["Feature"], importance_df["MeanAbsSHAP"])
plt.xlabel("Mean |SHAP value|")
plt.title("Feature importance based on SHAP values")
plt.gca().invert_yaxis()
plt.show()
