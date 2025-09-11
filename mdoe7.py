import pandas as pd
import numpy as np
import datetime
from xgboost import XGBRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error

# -------------------------
# Charger les CSV
# -------------------------
df_train = pd.read_csv("train_weather_processed.csv")
df_test = pd.read_csv("val_weather_processed.csv")

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

# Colonnes attractions
attraction_cols = [c for c in df_train.columns if c.startswith('ENTITY_DESCRIPTION_SHORT_')]

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
# Fonction pour créer des features agrégées des autres attractions
# -------------------------
def create_cross_attraction_features(df, current_attraction):
    """
    Crée des features agrégées basées sur les autres attractions
    """
    other_attractions = [att for att in attractions if att != current_attraction]
    cross_features = {}
    
    for att in other_attractions:
        att_col = f'ENTITY_DESCRIPTION_SHORT_{att}'
        
        # Moyenne des temps d'attente des autres attractions
        cross_features[f'mean_wait_time_{att}'] = df['CURRENT_WAIT_TIME'] * df[att_col]
        
        # Capacité ajustée des autres attractions
        cross_features[f'adj_capacity_{att}'] = df['ADJUST_CAPACITY'] * df[att_col]
    
    return pd.DataFrame(cross_features)

# -------------------------
# Splits 80/20 partagés entre toutes les attractions
# -------------------------
splits = ShuffleSplit(n_splits=5, test_size=0.07, random_state=42)
rmse_results = []
models_per_attraction = {}

for split_id, (train_idx, val_idx) in enumerate(splits.split(df_train), start=1):
    print(f"\n🔀 Split {split_id}/5")

    for att in attractions:
        att_col = f'ENTITY_DESCRIPTION_SHORT_{att}'

        # Filtrer uniquement les lignes de l'attraction
        df_att = df_train[df_train[att_col] == 1].reset_index(drop=False)

        if df_att.empty:
            continue  # attraction trop rare

        # Créer les features cross-attraction
        cross_features = create_cross_attraction_features(df_att, att)
        
        # Features globales et locales
        X_g = df_att[global_features]
        X_l = df_att[local_features + [att_col]]  # Inclure la colonne de l'attraction courante
        
        # Combiner toutes les features
        X = pd.concat([X_g, X_l, cross_features], axis=1)

        # Restreindre train_idx / val_idx aux indices globaux de df_att
        att_indices = df_att["index"].values
        train_mask = np.isin(att_indices, train_idx)
        val_mask   = np.isin(att_indices, val_idx)

        X_train, y_train = X[train_mask], df_att[target][train_mask]
        X_val, y_val = X[val_mask], df_att[target][val_mask]

        if len(y_train) == 0 or len(y_val) == 0:
            continue  # si split vide

        # Modèle XGBoost avec paramètres optimisés
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42 + split_id,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

        rmse_results.append({
            "Split": split_id,
            "Attraction": att,
            "RMSE": rmse,
            "Train_Size": len(y_train),
            "Val_Size": len(y_val)
        })

        # On conserve le dernier modèle entraîné pour la prédiction finale
        models_per_attraction[att] = model

        print(f"🎢 {att} (Split {split_id}) → RMSE = {rmse:.2f}, Train: {len(y_train)}, Val: {len(y_val)}")

# -------------------------
# Sauvegarde des résultats
# -------------------------
df_rmse = pd.DataFrame(rmse_results)
df_rmse.to_csv("rmse_per_attraction_splits_cross_features.csv", index=False)

print("\n✅ Fichier créé : rmse_per_attraction_splits_cross_features.csv")

# Moyenne par split
mean_rmse_by_split = df_rmse.groupby("Split")["RMSE"].mean()
print("\n📊 RMSE moyen par split :")
print(mean_rmse_by_split)

# RMSE moyen global
global_mean_rmse = df_rmse["RMSE"].mean()
print(f"\n🌍 RMSE moyen global sur tous les splits et attractions = {global_mean_rmse:.2f}")

# Analyse détaillée par attraction
attraction_stats = df_rmse.groupby("Attraction").agg({
    'RMSE': ['mean', 'std', 'count'],
    'Train_Size': 'sum',
    'Val_Size': 'sum'
}).round(2)

print("\n📈 Statistiques par attraction :")
print(attraction_stats)

# -------------------------
# Prédictions finales sur le test set
# -------------------------
predictions = []

for att in attractions:
    att_col = f'ENTITY_DESCRIPTION_SHORT_{att}'
    idx_test = df_test[att_col] == 1
    df_att_test = df_test[idx_test].reset_index(drop=True)

    if df_att_test.empty or att not in models_per_attraction:
        continue

    # Features cross-attraction
    cross_features_test = create_cross_attraction_features(df_att_test, att)

    # Features globales et locales
    X_test_g = df_att_test[global_features]
    X_test_l = df_att_test[local_features + [att_col]]
    
    X_test = pd.concat([X_test_g, X_test_l, cross_features_test], axis=1)

    model = models_per_attraction[att]
    y_pred = model.predict(X_test)

    for i, pred in zip(df_att_test.index, y_pred):
        row = df_att_test.loc[i]
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
df_pred.to_csv("predictions_val_weather_cross_features.csv", index=False)
print("\n✅ Fichier de prédictions créé : predictions_val_weather_cross_features.csv")