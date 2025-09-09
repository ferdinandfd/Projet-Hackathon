import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path, is_train=True, le=None):
    """
    Charge et prétraite les données du fichier CSV
    """
    # Charger les données
    df = pd.read_csv(file_path)
    
    # Convertir la colonne DATE_TIME en datetime
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    
    # Extraire des features temporelles
    df['YEAR'] = df['DATETIME'].dt.year
    df['MONTH'] = df['DATETIME'].dt.month
    df['DAY'] = df['DATETIME'].dt.day
    df['HOUR'] = df['DATETIME'].dt.hour
    df['MINUTE'] = df['DATETIME'].dt.minute
    df['DAY_OF_WEEK'] = df['DATETIME'].dt.dayofweek
    df['IS_WEEKEND'] = df['DAY_OF_WEEK'].isin([5, 6]).astype(int)
    
    # Gérer les valeurs manquantes
    parade_cols = ['TIME_TO_PARADE_1', 'TIME_TO_PARADE_2', 'TIME_TO_NIGHT_SHOW']
    for col in parade_cols:
        if col in df.columns:
            df[col] = df[col].fillna(-999)
    
    # Encoder la variable catégorielle ENTITY_DESCRIPTION_SHORT
    if le is None:
        le = LabelEncoder()
        df['ENTITY_ENCODED'] = le.fit_transform(df['ENTITY_DESCRIPTION_SHORT'])
    else:
        df['ENTITY_ENCODED'] = le.transform(df['ENTITY_DESCRIPTION_SHORT'])
    
    # Sélectionner les features pour le modèle
    feature_columns = [
        'ADJUST_CAPACITY', 'DOWNTIME', 'CURRENT_WAIT_TIME',
        'TIME_TO_PARADE_1', 'TIME_TO_PARADE_2', 'TIME_TO_NIGHT_SHOW',
        'YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'DAY_OF_WEEK', 'IS_WEEKEND',
        'ENTITY_ENCODED'
    ]
    
    # S'assurer que toutes les colonnes nécessaires existent
    available_features = [col for col in feature_columns if col in df.columns]
    
    X = df[available_features]
    
    if is_train:
        y = df['WAIT_TIME_IN_2H']
        return X, y, df, le, available_features
    else:
        return X, df, le, available_features

def train_model(X_train, y_train):
    """
    Entraîne un modèle de prédiction
    """
    # Imputer les valeurs manquantes
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    
    # Normaliser les features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    
    # Entraîner un modèle Random Forest
    model = RandomForestRegressor(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1,
        max_depth=10
    )
    
    model.fit(X_train_scaled, y_train)
    
    return model, imputer, scaler

def predict_validation_data(model, imputer, scaler, X_validation, validation_df, le):
    """
    Prépare les données de validation avec les prédictions
    """
    # Appliquer le même prétraitement
    X_validation_imputed = imputer.transform(X_validation)
    X_validation_scaled = scaler.transform(X_validation_imputed)
    
    # Faire les prédictions
    predictions = model.predict(X_validation_scaled)
    
    # Créer le dataframe de résultat
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
    train_file_path = 'waiting_times_train.csv'  # Fichier d'entraînement
    validation_file_path = 'waiting_times_X_test_val.csv'  # Fichier de validation
    
    try:
        # Charger et prétraiter les données d'entraînement
        print("Chargement des données d'entraînement...")
        X_train, y_train, train_df, le, feature_names = load_and_preprocess_data(train_file_path, is_train=True)
        
        print(f"Train dataset shape: {X_train.shape}")
        
        # Entraîner le modèle
        print("Entraînement du modèle...")
        model, imputer, scaler = train_model(X_train, y_train)
        
        # Charger et prétraiter les données de validation
        print("Chargement des données de validation...")
        X_validation, validation_df, le, feature_names = load_and_preprocess_data(
            validation_file_path, is_train=False, le=le
        )
        
        print(f"Validation dataset shape: {X_validation.shape}")
        
        # Faire les prédictions sur la validation
        print("Génération des prédictions...")
        validation_results = predict_validation_data(model, imputer, scaler, X_validation, validation_df, le)
        
        # Sauvegarder les résultats en CSV
        output_file = 'validation_predictions.csv'
        validation_results.to_csv(output_file, index=False)
        print(f"Fichier de prédictions sauvegardé sous : {output_file}")
        
        # Aperçu des prédictions
        print("\nAperçu des prédictions :")
        print(validation_results.head())
        
        print("Processus terminé avec succès!")
        
    except Exception as e:
        print(f"Une erreur s'est produite: {str(e)}")
        print("Assurez-vous que les chemins des fichiers sont corrects")

if __name__ == "__main__":
    main()