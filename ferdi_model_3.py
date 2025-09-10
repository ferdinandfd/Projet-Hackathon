import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

# -------------------------
# Charger les CSV
# -------------------------
df_train = pd.read_csv("train_weather_processed.csv")
df_val = pd.read_csv("val_weather_processed.csv")

# -------------------------
# Ajouter Flying Coaster si les autres attractions sont 0
# -------------------------
def add_flying_coaster(df):
    df['ENTITY_DESCRIPTION_SHORT_Flying_Coaster'] = (
        (df.get('ENTITY_DESCRIPTION_SHORT_Pirate Ship', 0) == 0) &
        (df.get('ENTITY_DESCRIPTION_SHORT_Water Ride', 0) == 0)
    ).astype(int)
    return df

df_train = add_flying_coaster(df_train)
df_val = add_flying_coaster(df_val)

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
    'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
    'CURRENT_WAIT_TIME'
]

# Ajouter toutes les colonnes one-hot pour les attractions
attraction_cols = [c for c in df_train.columns if c.startswith('ENTITY_DESCRIPTION_SHORT_')]
local_features += attraction_cols

# Ajouter les features météo disponibles
weather_features = ['feels_like', 'wind_speed', 'rain_1h', 'clouds_all']
local_features += weather_features

# Target
target = 'WAIT_TIME_IN_2H'

# -------------------------
# Traiter les inf pour les temps d'événement
# -------------------------
event_cols = ['TIME_TO_PARADE_1', 'TIME_TO_PARADE_2', 'TIME_TO_NIGHT_SHOW']
for col in event_cols:
    df_train[col] = df_train[col].replace(np.inf, 1e6)
    df_val[col] = df_val[col].replace(np.inf, 1e6)

# -------------------------
# Extraire la liste des attractions
# -------------------------
attractions = [c.replace('ENTITY_DESCRIPTION_SHORT_', '') for c in attraction_cols]

# -------------------------
# Choisir le meilleur modèle par attraction
# -------------------------
models_per_attraction = {}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for att in attractions:
    att_col = f'ENTITY_DESCRIPTION_SHORT_{att}'
    idx_train = df_train[att_col] == 1
    X_g = df_train.loc[idx_train, global_features]
    X_l = df_train.loc[idx_train, local_features]
    y = df_train.loc[idx_train, target]

    best_model = None
    best_rmse = float("inf")
    best_name = None

    # On teste 2 modèles : LinReg et XGB
    candidates = {
        "LinearRegression": LinearRegression(),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    }

    for name, model in candidates.items():
        rmses = []
        for train_idx, val_idx in kf.split(X_l):
            # On combine global et local pour chaque split
            X_train = pd.concat([X_g.iloc[train_idx].reset_index(drop=True),
                                 X_l.iloc[train_idx].reset_index(drop=True)], axis=1)
            X_val = pd.concat([X_g.iloc[val_idx].reset_index(drop=True),
                               X_l.iloc[val_idx].reset_index(drop=True)], axis=1)
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            rmses.append(np.sqrt(mean_squared_error(y_val, y_pred)))

        mean_rmse = np.mean(rmses)

        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_model = model
            best_name = name

    models_per_attraction[att] = best_model
    print(f"Attraction: {att}, meilleur modèle: {best_name}, RMSE CV: {best_rmse:.2f}")

print("\nMeilleurs modèles par attraction sélectionnés !")

# -------------------------
# Prédictions sur le fichier de validation
# -------------------------
predictions = []

for att in attractions:
    att_col = f'ENTITY_DESCRIPTION_SHORT_{att}'
    idx_val = df_val[att_col] == 1
    X_val_g = df_val.loc[idx_val, global_features].reset_index(drop=True)
    X_val_l = df_val.loc[idx_val, local_features].reset_index(drop=True)
    X_val = pd.concat([X_val_g, X_val_l], axis=1)

    model = models_per_attraction[att]
    y_pred = model.predict(X_val)

    for i, pred in zip(df_val.loc[idx_val].index, y_pred):
        row = df_val.loc[i]
        dt = datetime.datetime(
            int(row['year']), int(row['month']), int(row['day']),
            int(row['hour']), int(row['minute']), int(row['second'])
        )
        predictions.append({
            'DATETIME': dt,
            'ENTITY_DESCRIPTION_SHORT': att,
            'y_pred': pred,
            'KEY': 'Validation'
        })

# Convertir en DataFrame et sauvegarder
df_pred = pd.DataFrame(predictions)
df_pred.to_csv("predictions_val_weather_3.csv", index=False)
print("\nFichier de prédictions créé : predictions_val_weather.csv")
