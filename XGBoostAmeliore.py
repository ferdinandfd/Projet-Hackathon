import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

def advanced_feature_engineering(df, le=None):
    """
    Feature engineering avancé
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
    
    # Encoder les entités
    if le is None:
        le = LabelEncoder()
        df['ENTITY_ENCODED'] = le.fit_transform(df['ENTITY_DESCRIPTION_SHORT'])
    else:
        df['ENTITY_ENCODED'] = le.transform(df['ENTITY_DESCRIPTION_SHORT'])
    
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
    
    return df, le

def prepare_features(df):
    """
    Préparation des features pour le modèle
    """
    # Features à utiliser
    feature_columns = [
        'ADJUST_CAPACITY', 'DOWNTIME', 'CURRENT_WAIT_TIME',
        'TIME_TO_PARADE_1', 'TIME_TO_PARADE_2', 'TIME_TO_NIGHT_SHOW',
        'YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'DAY_OF_WEEK', 
        'IS_WEEKEND', 'WEEK_OF_YEAR', 'HOUR_SIN', 'HOUR_COS',
        'SEASON', 'ENTITY_ENCODED', 'CAPACITY_WAIT_RATIO', 'WAIT_TIME_TREND', 'dew_point', 'feels,like', 'humidity', 'wind_speed', 'rain_1h', 'clouds_all'
    ]
    
    # Garder seulement les colonnes disponibles
    available_features = [col for col in feature_columns if col in df.columns]
    X = df[available_features]
    
    return X, available_features

def train_optimized_model(X_train, y_train):
    """
    Entraînement d'un modèle optimisé
    """
    # Imputation simple mais efficace
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    
    # Modèle Gradient Boosting optimisé
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=4,
        subsample=0.8,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    
    # Validation croisée
    print("Validation croisée en cours...")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                               cv=3, scoring='neg_root_mean_squared_error')
    print(f"RMSE cross-validation: {-cv_scores.mean():.2f} (±{-cv_scores.std():.2f})")
    
    # Entraînement final
    print("Entraînement final du modèle...")
    model.fit(X_train_scaled, y_train)
    
    return model, imputer, scaler

def predict_validation_set(model, imputer, scaler, X_validation, validation_df):
    """
    Prédiction sur l'ensemble de validation
    """
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
    # Chemins des fichiers - À MODIFIER !
    train_file_path = 'waiting_times_train.csv'
    validation_file_path = 'waiting_times_X_test_val.csv'
    
    try:
        print("1. Chargement des données d'entraînement...")
        df1 = pd.read_csv(train_file_path)
        dfweather = pd.read_csv('weather_data.csv')
        train_df = pd.merge(df1, dfweather)

        
        print("2. Feature engineering...")
        train_df, le = advanced_feature_engineering(train_df)
        
        print("3. Préparation des features...")
        X_train, feature_names = prepare_features(train_df)
        y_train = train_df['WAIT_TIME_IN_2H']
        
        print(f"Shape: {X_train.shape}")
        print(f"Target range: {y_train.min():.1f} - {y_train.max():.1f}")
        
        print("4. Entraînement du modèle optimisé...")
        model, imputer, scaler = train_optimized_model(X_train, y_train)
        
        print("5. Chargement des données de validation...")
        df2 = pd.read_csv(validation_file_path)
        validation_df = pd.merge(df2, dfweather)
        validation_df, _ = advanced_feature_engineering(validation_df, le)
        
        print("6. Préparation des features de validation...")
        X_validation, _ = prepare_features(validation_df)
        
        print("7. Génération des prédictions...")
        results = predict_validation_set(model, imputer, scaler, X_validation, validation_df)
        
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
        print(f"❌ Erreur: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()