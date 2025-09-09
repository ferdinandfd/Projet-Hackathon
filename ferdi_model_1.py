import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# -------------------------
# Charger les CSV
# -------------------------
df_train = pd.read_csv("waiting_times_train_preprocessed.csv")
df_val = pd.read_csv("waiting_times_X_test_val_preprocessed.csv")

# -------------------------
# Définir features globales et locales
# -------------------------
global_features = [
    'year', 'month', 'day', 'day_of_week', 'is_weekend',
    'is_there_parade_1', 'is_there_parade_2', 'is_there_night_show'
]

local_features = [
    'ADJUST_CAPACITY', 'DOWNTIME',
    'TIME_TO_PARADE_1', 'TIME_TO_PARADE_2', 'TIME_TO_NIGHT_SHOW',
    'hour', 'minute', 'second',
    'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
    'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos'
]

# Ajouter toutes les colonnes one-hot des attractions
attraction_cols = [c for c in df_train.columns if c.startswith('ENTITY_DESCRIPTION_SHORT_')]
local_features += attraction_cols

# Target
target = 'CURRENT_WAIT_TIME'

# -------------------------
# Préparer X et y
# -------------------------
X_train_global = df_train[global_features]
X_train_local = df_train[local_features]
y_train = df_train[target]

X_val_global = df_val[global_features]
X_val_local = df_val[local_features]
y_val = df_val[target]

# -------------------------
# Entraîner les modèles sur tout le train
# -------------------------
model_global = LinearRegression()
model_global.fit(X_train_global, y_train)

model_local = LinearRegression()
model_local.fit(X_train_local, y_train)

# -------------------------
# Cross-validation interne pour ajuster la pondération
# -------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
weights = np.linspace(0, 1, 21)
best_weight = 0
best_rmse_cv = float('inf')

for w in weights:
    rmses = []
    for train_idx, val_idx in kf.split(X_train_global):
        # Split interne
        X_tr_g, X_val_g = X_train_global.iloc[train_idx], X_train_global.iloc[val_idx]
        X_tr_l, X_val_l = X_train_local.iloc[train_idx], X_train_local.iloc[val_idx]
        y_tr, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Prédictions modèles déjà entraînés
        y_pred_val = w * model_global.predict(X_val_g) + (1 - w) * model_local.predict(X_val_l)
        rmses.append(np.sqrt(mean_squared_error(y_val_cv, y_pred_val)))
    
    mean_rmse = np.mean(rmses)
    if mean_rmse < best_rmse_cv:
        best_rmse_cv = mean_rmse
        best_weight = w

print(f"Meilleure pondération (CV sur train): {best_weight:.2f}, RMSE CV: {best_rmse_cv:.2f}")

# -------------------------
# Évaluation finale sur le set de validation
# -------------------------
y_pred_val_combined = best_weight * model_global.predict(X_val_global) + (1 - best_weight) * model_local.predict(X_val_local)
rmse_final = np.sqrt(mean_squared_error(y_val, y_pred_val_combined))
print(f"RMSE final sur validation: {rmse_final:.2f}")
