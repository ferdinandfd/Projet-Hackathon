import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, RepeatedKFold
import joblib
import warnings
import optuna
import xgboost as xgb
from xgboost import XGBRegressor
warnings.filterwarnings('ignore')

def advanced_feature_engineering(df, entity_columns=None):
    """
    Feature engineering avancé avec One-Hot Encoding
    """
    df = df.copy()
    
    # Conversion datetime
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    
    # Features temporelles de base
    df['YEAR'] = df['DATETIME'].dt.year
    df['MONTH'] = df['DATETIME'].dt.month
    df['DAY'] = df['DATETIME'].dt.day
    df['HOUR'] = df['DATETIME'].dt.hour
    df['MINUTE'] = df['DATETIME'].dt.minute
    df['DAY_OF_WEEK'] = df['DATETIME'].dt.dayofweek
    df['IS_WEEKEND'] = df['DAY_OF_WEEK'].isin([5, 6]).astype(int)
    df['WEEK_OF_YEAR'] = df['DATETIME'].dt.isocalendar().week
    
    # Features cycliques pour l'heure
    df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24)
    df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24)
    
    # Saisonnalité
    df['SEASON'] = (df['MONTH'] % 12 + 3) // 3
    
    # One-Hot Encoding pour les entités
    if entity_columns is None:
        # Première fois - créer les colonnes one-hot
        entity_dummies = pd.get_dummies(df['ENTITY_DESCRIPTION_SHORT'], prefix='ENTITY')
        df = pd.concat([df, entity_dummies], axis=1)
        entity_columns = entity_dummies.columns.tolist()
    else:
        # Utiliser les colonnes existantes et ajouter 0 si manquant
        for col in entity_columns:
            if col not in df.columns:
                df[col] = 0
    
    # Interactions entre variables
    if 'CURRENT_WAIT_TIME' in df.columns and 'ADJUST_CAPACITY' in df.columns:
        df['CAPACITY_WAIT_RATIO'] = df['CURRENT_WAIT_TIME'] / (df['ADJUST_CAPACITY'] + 1)
        df['WAIT_TIME_TREND'] = df['CURRENT_WAIT_TIME'] - df['ADJUST_CAPACITY']
    
    # Gestion des valeurs manquantes pour les parades
    parade_cols = ['TIME_TO_PARADE_1', 'TIME_TO_PARADE_2', 'TIME_TO_NIGHT_SHOW']
    for col in parade_cols:
        if col in df.columns:
            # Remplacer les NaN par la médiane de la colonne
            if df[col].notna().sum() > 0:  # S'il y a des valeurs non-nulles
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
            else:
                df[col] = df[col].fillna(-999)
    
    return df, entity_columns

def prepare_features(df):
    """
    Préparation des features pour le modèle
    """
    # Features à utiliser (uniquement numériques)
    feature_columns = [
        'ADJUST_CAPACITY', 'DOWNTIME', #'CURRENT_WAIT_TIME',
        'TIME_TO_PARADE_1', 'TIME_TO_PARADE_2', #'TIME_TO_NIGHT_SHOW',
        'YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'DAY_OF_WEEK', 
        'IS_WEEKEND', 'WEEK_OF_YEAR', 'HOUR_SIN', 'HOUR_COS',
        'SEASON', 'CAPACITY_WAIT_RATIO', 'WAIT_TIME_TREND', 
        'dew_point', 'feels_like', 'humidity', 'wind_speed', 'rain_1h', 'clouds_all'
    ]
    
    # Ajouter automatiquement les colonnes one-hot (commencent par 'ENTITY_')
    one_hot_columns = [col for col in df.columns if col.startswith('ENTITY_')]
    feature_columns.extend(one_hot_columns)
    
    # Garder seulement les colonnes disponibles ET numériques
    available_features = [col for col in feature_columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    # Vérifier s'il y a des colonnes non-numériques et les exclure
    non_numeric_cols = [col for col in feature_columns if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric_cols:
        print(f"⚠️  Colonnes non-numériques exclues: {non_numeric_cols}")
    
    X = df[available_features]
    
    return X, available_features

def train_optimized_model(X_train, y_train):
    # Préprocessing
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    train_columns = X_train.columns.tolist()

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 600),  # plus d'arbres
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.08),
            "max_depth": trial.suggest_int("max_depth", 5, 8),  # un peu plus profond
            "subsample": trial.suggest_uniform("subsample", 0.7, 0.85),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.7, 0.85),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_uniform("gamma", 0, 5),
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 1.0),
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 1.0),
            "objective": "reg:squarederror",
            "n_jobs": -1,
            "random_state": 42
        }

        model = xgb.XGBRegressor(**params)

        # RepeatedKFold pour plus de stabilité
        rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
        scores = cross_val_score(
            model, 
            X_train_scaled, 
            y_train,
            cv=rkf,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )
        return -np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)  # peut augmenter à 70–80 si le temps le permet

    print("✅ Best RMSE CV:", study.best_value)
    print("✅ Best params:", study.best_trial.params)

    # Entraînement final sur tout le train
    best_params = study.best_trial.params
    best_params["objective"] = "reg:squarederror"
    best_params["n_jobs"] = -1
    best_params["random_state"] = 42

    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(
        X_train_scaled,
        y_train,
        eval_set=[(X_train_scaled, y_train)],
        early_stopping_rounds=50,
        verbose=True
    )

def train_bis(X_train, y_train):
    """
    Entraîne un modèle XGBoost avec des paramètres prédéfinis (Optuna déjà effectué).
    Retourne le modèle, l'imputer, le scaler et les colonnes utilisées.
    """

    # 1️⃣ Imputation des valeurs manquantes
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)

    # 2️⃣ Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)

    # 3️⃣ Colonnes
    train_columns = X_train.columns.tolist()

    # 4️⃣ Meilleurs paramètres déjà trouvés
    best_params = {
        'n_estimators': 559,
        'learning_rate': 0.06850817756224654,
        'max_depth': 8,
        'subsample': 0.7296827274618055,
        'colsample_bytree': 0.8042790364961063,
        'min_child_weight': 6,
        'gamma': 1.4098901770042573,
        'reg_alpha': 2.621011296818228e-07,
        'reg_lambda': 6.437379565634366e-07,
        'objective': 'reg:squarederror',
        'n_jobs': -1,
        'random_state': 42
    }

    # 5️⃣ Entraînement du modèle
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train_scaled, y_train)  # sans early stopping

    return model, imputer, scaler, train_columns



    return final_model, imputer, scaler, train_columns

def predict_validation_set(model, imputer, scaler, X_validation, validation_df, train_columns):
    """
    Prédiction sur l'ensemble de validation
    """
    # S'assurer que X_validation a les mêmes colonnes que X_train
    missing_cols = set(train_columns) - set(X_validation.columns)
    extra_cols = set(X_validation.columns) - set(train_columns)
    
    # Ajouter les colonnes manquantes avec 0
    for col in missing_cols:
        X_validation[col] = 0
    
    # Supprimer les colonnes en trop
    X_validation = X_validation[train_columns]
    
    # Vérifier à nouveau les types numériques
    X_validation = X_validation.select_dtypes(include=[np.number])
    
    # Application du préprocessing
    X_val_imputed = imputer.transform(X_validation)
    X_val_scaled = scaler.transform(X_val_imputed)
    
    # Prédictions
    predictions = model.predict(X_val_scaled)
    
    # Assurer que les prédictions sont réalistes
    predictions = np.clip(predictions, 0, 120)  # Limiter entre 0 et 120 minutes
    
    # DataFrame de résultats
    result_df = pd.DataFrame({
        'DATETIME': validation_df['DATETIME'],
        'ENTITY_DESCRIPTION_SHORT': validation_df['ENTITY_DESCRIPTION_SHORT'],
        'y_pred': predictions,
        'KEY': 'Validation'
    })
    
    return result_df

def main():
    """
    Fonction principale
    """
    # Chemins des fichiers
    train_file_path = 'waiting_times_train.csv'
    validation_file_path = 'waiting_times_X_test_val.csv'
    
    try:
        print("1. Chargement des données d'entraînement...")
        df1 = pd.read_csv(train_file_path)
        dfweather = pd.read_csv('weather_data.csv')
        train_df = pd.merge(df1, dfweather)

        print("2. Feature engineering avec One-Hot Encoding...")
        train_df, entity_columns = advanced_feature_engineering(train_df)
        
        print("3. Préparation des features...")
        X_train, feature_names = prepare_features(train_df)
        y_train = train_df['WAIT_TIME_IN_2H']
        
        print(f"Shape: {X_train.shape}")
        print(f"Nombre de features: {len(feature_names)}")
        print(f"Target range: {y_train.min():.1f} - {y_train.max():.1f}")
        
        print("4. Entraînement du modèle optimisé...")
        model, imputer, scaler, train_columns = train_bis(X_train, y_train)
        
        print("5. Chargement des données de validation...")
        df2 = pd.read_csv(validation_file_path)
        validation_df = pd.merge(df2, dfweather)
        validation_df, _ = advanced_feature_engineering(validation_df, entity_columns)
        
        print("6. Préparation des features de validation...")
        X_validation, _ = prepare_features(validation_df)
        
        print("7. Génération des prédictions...")
        results = predict_validation_set(model, imputer, scaler, X_validation, validation_df, train_columns)
        
        # Validation des prédictions
        print(f"\n📊 Statistiques des prédictions:")
        print(f"Min: {results['y_pred'].min():.2f} minutes")
        print(f"Max: {results['y_pred'].max():.2f} minutes")
        print(f"Moyenne: {results['y_pred'].mean():.2f} minutes")
        print(f"Écart-type: {results['y_pred'].std():.2f} minutes")
        
        # Sauvegarde
        output_file = 'validation_predictions_optimized.csv'
        results.to_csv(output_file, index=False)
        print(f"\n💾 Fichier sauvegardé: {output_file}")
        
        print("\n✅ Processus terminé avec succès!")
        
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()