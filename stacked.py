import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path, is_train=True, le=None):
    """
    Charge et prétraite les données du fichier CSV
    """
    # Charger les données
    df1 = pd.read_csv(file_path)
    dfweather = pd.read_csv('weather_data.csv')

    df = pd.merge(df1, dfweather, how='left')
    
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
    
    # Features supplémentaires
    df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24)
    df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24)
    df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH'] / 12)
    df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH'] / 12)
    
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
        # Gérer les nouvelles catégories dans le dataset de test
        df['ENTITY_ENCODED'] = df['ENTITY_DESCRIPTION_SHORT'].apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
    
    # Sélectionner les features pour le modèle
    feature_columns = [
        'ADJUST_CAPACITY', 'DOWNTIME', 'CURRENT_WAIT_TIME',
        'TIME_TO_PARADE_1', 'TIME_TO_PARADE_2', 'TIME_TO_NIGHT_SHOW',
        'YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'DAY_OF_WEEK', 'IS_WEEKEND',
        'ENTITY_ENCODED', 'HOUR_SIN', 'HOUR_COS', 'MONTH_SIN', 'MONTH_COS'
    ]
    
    # Ajouter les features météo si elles existent
    weather_features = [col for col in df.columns if col.startswith(('TEMP', 'HUMIDITY', 'RAIN', 'WIND'))]
    feature_columns.extend(weather_features)
    
    # S'assurer que toutes les colonnes nécessaires existent
    available_features = [col for col in feature_columns if col in df.columns]
    
    X = df[available_features]
    
    if is_train:
        y = df['WAIT_TIME_IN_2H']
        return X, y, df, le, available_features
    else:
        return X, df, le, available_features

class StackingRegressor:
    """
    Modèle de stacking personnalisé avec CatBoost, LightGBM et Random Forest
    """
    
    def __init__(self, n_folds=15, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.base_models = {}
        self.meta_model = None
        self.imputer = None
        self.scaler = None
        self.feature_names = None
        
    def _get_base_models(self):
        """
        Définit les modèles de base
        """
        models = {
            'rf': RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'lgb': lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1
            ),
            'catboost': cb.CatBoostRegressor(
                iterations=200,
                depth=8,
                learning_rate=0.1,
                l2_leaf_reg=3,
                random_seed=self.random_state,
                verbose=False
            ),

            'xgb' : xgb.XGBRegressor(
                n_estimators= 559,
                learning_rate = 0.06850817756224654,
                max_depth =  8,
                subsample = 0.7296827274618055,
                colsample_bytree = 0.8042790364961063,
                min_child_weight = 6,
                gamma = 1.4098901770042573,
                reg_alpha = 2.621011296818228e-07,
                reg_lambda = 6.437379565634366e-07,
                objective = 'reg:squarederror',
                n_jobs = -1,
                random_state = 42)
        }
        return models
    
    def fit(self, X, y):
        """
        Entraîne le modèle de stacking
        """
        # Prétraitement des données
        self.imputer = SimpleImputer(strategy='median')
        X_imputed = self.imputer.fit_transform(X)
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        self.feature_names = X.columns.tolist()
        
        # Initialiser les modèles de base
        base_models = self._get_base_models()
        
        # Créer les prédictions out-of-fold pour le stacking
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        oof_predictions = np.zeros((X_scaled.shape[0], len(base_models)))
        
        print("Entraînement des modèles de base avec validation croisée...")
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled)):
            print(f"Fold {fold + 1}/{self.n_folds}")
            
            X_train_fold = X_scaled[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X_scaled[val_idx]
            
            for i, (model_name, model) in enumerate(base_models.items()):
                # Cloner le modèle pour ce fold
                model_clone = self._clone_model(model)
                
                # Entraîner sur le fold
                model_clone.fit(X_train_fold, y_train_fold)
                
                # Prédire sur la validation
                oof_predictions[val_idx, i] = model_clone.predict(X_val_fold)
        
        # Entraîner les modèles finaux sur toutes les données
        print("Entraînement des modèles finaux...")
        for model_name, model in base_models.items():
            model.fit(X_scaled, y)
            self.base_models[model_name] = model
        
        # Entraîner le meta-modèle
        self.meta_model = LinearRegression()
        self.meta_model.fit(oof_predictions, y)
        
        # Calculer les métriques de validation croisée
        oof_final_pred = self.meta_model.predict(oof_predictions)
        cv_rmse = np.sqrt(mean_squared_error(y, oof_final_pred))
        cv_mae = mean_absolute_error(y, oof_final_pred)
        
        print(f"CV RMSE: {cv_rmse:.4f}")
        print(f"CV MAE: {cv_mae:.4f}")
        
        return self
    
    def _clone_model(self, model):
        """
        Clone un modèle avec les mêmes paramètres
        """
        if isinstance(model, RandomForestRegressor):
            return RandomForestRegressor(**model.get_params())
        elif isinstance(model, lgb.LGBMRegressor):
            return lgb.LGBMRegressor(**model.get_params())
        elif isinstance(model, cb.CatBoostRegressor):
            return cb.CatBoostRegressor(**model.get_params())
        elif isinstance(model, xgb.XGBRegressor):
            return xgb.XGBRegressor(**model.get_params())
        else:
            raise ValueError(f"Modèle non supporté: {type(model)}")
    
    def predict(self, X):
        """
        Fait des prédictions avec le modèle de stacking
        """
        # Prétraitement
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        
        # Prédictions des modèles de base
        base_predictions = np.zeros((X_scaled.shape[0], len(self.base_models)))
        
        for i, (model_name, model) in enumerate(self.base_models.items()):
            base_predictions[:, i] = model.predict(X_scaled)
        
        # Prédiction finale avec le meta-modèle
        final_predictions = self.meta_model.predict(base_predictions)
        
        return final_predictions

def train_stacking_model(X_train, y_train):
    """
    Entraîne le modèle de stacking
    """
    stacking_model = StackingRegressor(n_folds=5, random_state=42)
    stacking_model.fit(X_train, y_train)
    
    return stacking_model

def predict_validation_data(model, X_validation, validation_df):
    """
    Prépare les données de validation avec les prédictions
    """
    # Faire les prédictions
    predictions = model.predict(X_validation)
    
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
        print("=" * 60)
        print("MODÈLE DE STACKING AVEC CATBOOST, LIGHTGBM ET RANDOM FOREST")
        print("=" * 60)
        
        # Charger et prétraiter les données d'entraînement
        print("Chargement des données d'entraînement...")
        X_train, y_train, train_df, le, feature_names = load_and_preprocess_data(train_file_path, is_train=True)
        
        print(f"Train dataset shape: {X_train.shape}")
        print(f"Features utilisées: {len(feature_names)}")
        print(f"Nombre d'échantillons d'entraînement: {len(y_train)}")
        
        # Entraîner le modèle de stacking
        print("\n" + "=" * 40)
        print("ENTRAÎNEMENT DU MODÈLE DE STACKING")
        print("=" * 40)
        
        stacking_model = train_stacking_model(X_train, y_train)
        
        # Charger et prétraiter les données de validation
        print("\n" + "=" * 40)
        print("PRÉDICTION SUR LES DONNÉES DE VALIDATION")
        print("=" * 40)
        
        print("Chargement des données de validation...")
        X_validation, validation_df, le, feature_names = load_and_preprocess_data(
            validation_file_path, is_train=False, le=le
        )
        
        print(f"Validation dataset shape: {X_validation.shape}")
        
        # Faire les prédictions sur la validation
        print("Génération des prédictions...")
        validation_results = predict_validation_data(stacking_model, X_validation, validation_df)
        
        # Sauvegarder les résultats en CSV
        output_file = 'validation_predictions_stacking.csv'
        validation_results.to_csv(output_file, index=False)
        print(f"Fichier de prédictions sauvegardé sous : {output_file}")
        
        # Sauvegarder le modèle
        model_file = 'stacking_model.joblib'
        joblib.dump({
            'model': stacking_model,
            'label_encoder': le,
            'feature_names': feature_names
        }, model_file)
        print(f"Modèle sauvegardé sous : {model_file}")
        
        # Aperçu des prédictions
        print("\n" + "=" * 40)
        print("RÉSULTATS")
        print("=" * 40)
        
        print(f"Nombre de prédictions générées: {len(validation_results)}")
        print(f"Moyenne des prédictions: {validation_results['y_pred'].mean():.2f}")
        print(f"Médiane des prédictions: {validation_results['y_pred'].median():.2f}")
        print(f"Écart-type des prédictions: {validation_results['y_pred'].std():.2f}")
        
        print("\nAperçu des prédictions :")
        print(validation_results.head(10))
        
        print("\nStatistiques par entité :")
        entity_stats = validation_results.groupby('ENTITY_DESCRIPTION_SHORT')['y_pred'].agg(['count', 'mean', 'std']).round(2)
        print(entity_stats.head())
        
        print("\n" + "=" * 60)
        print("PROCESSUS TERMINÉ AVEC SUCCÈS!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nUne erreur s'est produite: {str(e)}")
        print("Assurez-vous que:")
        print("1. Les chemins des fichiers sont corrects")
        print("2. Les packages requis sont installés: xgboost, lightgbm, catboost, scikit-learn, pandas, numpy")
        print("3. Le fichier weather_data.csv existe et est accessible")
        import traceback
        print("\nDétails de l'erreur:")
        traceback.print_exc()

if __name__ == "__main__":
    main()