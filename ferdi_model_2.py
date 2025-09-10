import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

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
    'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'CURRENT_WAIT_TIME'
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
# Initialiser modèles globaux et locaux
# -------------------------
model_global = LinearRegression()
model_global.fit(df_train[global_features], df_train[target])

model_local = LinearRegression()
model_local.fit(df_train[local_features], df_train[target])

# -------------------------
# Extraire la liste des attractions
# -------------------------
attractions = [c.replace('ENTITY_DESCRIPTION_SHORT_', '') for c in attraction_cols]

# -------------------------
# Cross-validation interne pour chaque attraction
# -------------------------
weights_per_attraction = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for att in attractions:
    att_col = f'ENTITY_DESCRIPTION_SHORT_{att}'
    idx_train = df_train[att_col] == 1
    X_tr_g = df_train.loc[idx_train, global_features]
    X_tr_l = df_train.loc[idx_train, local_features]
    y_tr = df_train.loc[idx_train, target]
    
    best_weight = 0
    best_rmse = float('inf')
    
    for w in np.linspace(0, 1, 21):
        rmses = []
        for train_idx, val_idx in kf.split(X_tr_g):
            X_t_g, X_v_g = X_tr_g.iloc[train_idx], X_tr_g.iloc[val_idx]
            X_t_l, X_v_l = X_tr_l.iloc[train_idx], X_tr_l.iloc[val_idx]
            y_t, y_v = y_tr.iloc[train_idx], y_tr.iloc[val_idx]
            
            y_pred = w * model_global.predict(X_v_g) + (1 - w) * model_local.predict(X_v_l)
            rmses.append(np.sqrt(mean_squared_error(y_v, y_pred)))
        
        mean_rmse = np.mean(rmses)
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_weight = w
    
    weights_per_attraction[att] = best_weight
    print(f"Attraction: {att}, meilleure pondération: {best_weight:.2f}, RMSE CV: {best_rmse:.2f}")

print("\nPondérations global/local par attraction calculées avec succès !")

# -------------------------
# Prédictions sur le fichier de validation
# -------------------------
predictions = []

for att in attractions:
    att_col = f'ENTITY_DESCRIPTION_SHORT_{att}'
    idx_val = df_val[att_col] == 1
    X_v_g = df_val.loc[idx_val, global_features]
    X_v_l = df_val.loc[idx_val, local_features]
    w = weights_per_attraction[att]
    
    y_pred = w * model_global.predict(X_v_g) + (1 - w) * model_local.predict(X_v_l)
    
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
df_pred.to_csv("predictions_val_weather.csv", index=False)
print("\nFichier de prédictions créé : predictions_val_weather.csv")
