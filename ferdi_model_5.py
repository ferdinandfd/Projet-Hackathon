import pandas as pd
import numpy as np
import datetime
from xgboost import XGBRegressor

# -------------------------
# Charger les CSV
# -------------------------
df_train = pd.read_csv("train_weather_processed.csv")
df_test = pd.read_csv("final_weather_processed.csv")

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
df_test = add_flying_coaster(df_test)

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

# Ajouter les features météo
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
    df_test[col] = df_test[col].replace(np.inf, 1e6)

# -------------------------
# Extraire la liste des attractions
# -------------------------
attractions = [c.replace('ENTITY_DESCRIPTION_SHORT_', '') for c in attraction_cols]

# -------------------------
# Entraîner XGBoost profondeur 5, learning_rate 0.07
# -------------------------
models_per_attraction = {}

for att in attractions:
    att_col = f'ENTITY_DESCRIPTION_SHORT_{att}'
    idx_train = df_train[att_col] == 1
    X_g = df_train.loc[idx_train, global_features]
    X_l = df_train.loc[idx_train, local_features]
    y = df_train.loc[idx_train, target]

    model = XGBRegressor(
        n_estimators=295,
        learning_rate=0.06,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    X_train = pd.concat([X_g.reset_index(drop=True), X_l.reset_index(drop=True)], axis=1)
    model.fit(X_train, y)
    models_per_attraction[att] = model
    print(f"Attraction: {att}, modèle XGBoost entraîné.")

print("\nTous les modèles XGBoost par attraction entraînés !")

# -------------------------
# Prédictions sur le fichier test final
# -------------------------
predictions = []

for att in attractions:
    att_col = f'ENTITY_DESCRIPTION_SHORT_{att}'
    idx_test = df_test[att_col] == 1
    X_test_g = df_test.loc[idx_test, global_features].reset_index(drop=True)
    X_test_l = df_test.loc[idx_test, local_features].reset_index(drop=True)
    X_test = pd.concat([X_test_g, X_test_l], axis=1)

    model = models_per_attraction[att]
    y_pred = model.predict(X_test)
    #y_pred_min=np.round(y_pred/2)*2

    for i, pred in zip(df_test.loc[idx_test].index, y_pred):
        row = df_test.loc[i]
        dt = datetime.datetime(
            int(row['year']), int(row['month']), int(row['day']),
            int(row['hour']), int(row['minute']), int(row['second'])
        )
        predictions.append({
            'DATETIME': dt,
            'ENTITY_DESCRIPTION_SHORT': att,
            'y_pred': pred,
            'KEY': 'Final'
        })

# Convertir en DataFrame et sauvegarder
df_pred = pd.DataFrame(predictions)
df_pred.to_csv("predictions_val_final.csv", index=False)
print("\nFichier de prédictions créé : predictions_val_final.csv")
