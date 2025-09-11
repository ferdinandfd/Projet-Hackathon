import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

def advanced_feature_engineering(df, entity_columns=None):
    """
    Feature engineering avancé avec One-Hot Encoding
    """
    df = df.copy()
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    
    # Features temporelles
    df['YEAR'] = df['DATETIME'].dt.year
    df['MONTH'] = df['DATETIME'].dt.month
    df['DAY'] = df['DATETIME'].dt.day
    df['HOUR'] = df['DATETIME'].dt.hour
    df['MINUTE'] = df['DATETIME'].dt.minute
    df['DAY_OF_WEEK'] = df['DATETIME'].dt.dayofweek
    df['IS_WEEKEND'] = df['DAY_OF_WEEK'].isin([5, 6]).astype(int)
    df['WEEK_OF_YEAR'] = df['DATETIME'].dt.isocalendar().week
    
    # Variables cycliques
    df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24)
    df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24)
    df['SEASON'] = (df['MONTH'] % 12 + 3) // 3
    
    # One-hot encoding
    if entity_columns is None:
        entity_dummies = pd.get_dummies(df['ENTITY_DESCRIPTION_SHORT'], prefix='ENTITY')
        df = pd.concat([df, entity_dummies], axis=1)
        entity_columns = entity_dummies.columns.tolist()
    else:
        for col in entity_columns:
            if col not in df.columns:
                df[col] = 0
    
    # Interactions
    if 'CURRENT_WAIT_TIME' in df.columns and 'ADJUST_CAPACITY' in df.columns:
        df['CAPACITY_WAIT_RATIO'] = df['CURRENT_WAIT_TIME'] / (df['ADJUST_CAPACITY'] + 1)
        df['WAIT_TIME_TREND'] = df['CURRENT_WAIT_TIME'] - df['ADJUST_CAPACITY']
    
    # Gestion des valeurs manquantes
    for col in ['TIME_TO_PARADE_1', 'TIME_TO_PARADE_2', 'TIME_TO_NIGHT_SHOW']:
        if col in df.columns:
            if df[col].notna().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(-999)
    
    return df, entity_columns

def prepare_features(df):
    """
    Sélection et préparation des features
    """
    base_features = [
        'ADJUST_CAPACITY', 'DOWNTIME', 'CURRENT_WAIT_TIME',
        'TIME_TO_PARADE_1', 'TIME_TO_PARADE_2', 'TIME_TO_NIGHT_SHOW',
        'YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'DAY_OF_WEEK', 
        'IS_WEEKEND', 'WEEK_OF_YEAR', 'HOUR_SIN', 'HOUR_COS',
        'SEASON', 'CAPACITY_WAIT_RATIO', 'WAIT_TIME_TREND',
        'dew_point', 'feels_like', 'humidity', 'wind_speed', 'rain_1h', 'clouds_all'
    ]
    one_hot_columns = [col for col in df.columns if col.startswith('ENTITY_')]
    feature_columns = base_features + one_hot_columns
    
    available = [c for c in feature_columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    return df[available], available

def train_optimized_model(X_train, y_train):
    """
    Entraînement avec AdaBoostClassifier
    """
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    
    # Modèle AdaBoost
    base_tree = DecisionTreeClassifier(max_depth=3)
    model = AdaBoostClassifier(
        base_estimator=base_tree,
        n_estimators=300,
        learning_rate=0.5,
        random_state=42
    )
    
    # Validation croisée (sur RMSE)
    print("Validation croisée...")
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train,
        cv=3, scoring='neg_root_mean_squared_error'
    )
    print(f"RMSE CV: {-cv_scores.mean():.2f} (±{-cv_scores.std():.2f})")
    
    # Entraînement final
    model.fit(X_train_scaled, y_train)
    
    return model, imputer, scaler, X_train.columns.tolist()

def predict_validation_set(model, imputer, scaler, X_val, validation_df, train_columns):
    """
    Prédiction sur le set de validation
    """
    for col in set(train_columns) - set(X_val.columns):
        X_val[col] = 0
    X_val = X_val[train_columns]
    
    X_val_imputed = imputer.transform(X_val)
    X_val_scaled = scaler.transform(X_val_imputed)
    
    preds = model.predict(X_val_scaled)
    
    result_df = pd.DataFrame({
        'DATETIME': validation_df['DATETIME'],
        'ENTITY_DESCRIPTION_SHORT': validation_df['ENTITY_DESCRIPTION_SHORT'],
        'y_pred': preds,
        'KEY': 'Validation'
    })
    return result_df

def main():
    train_file = 'waiting_times_train.csv'
    validation_file = 'waiting_times_X_test_val.csv'
    weather_file = 'weather_data.csv'
    
    df_train = pd.read_csv(train_file)
    df_weather = pd.read_csv(weather_file)
    train_df = pd.merge(df_train, df_weather)
    
    train_df, entity_cols = advanced_feature_engineering(train_df)
    X_train, features = prepare_features(train_df)
    y_train = train_df['WAIT_TIME_IN_2H']
    
    model, imputer, scaler, train_cols = train_optimized_model(X_train, y_train)
    
    df_val = pd.read_csv(validation_file)
    val_df = pd.merge(df_val, df_weather)
    val_df, _ = advanced_feature_engineering(val_df, entity_cols)
    X_val, _ = prepare_features(val_df)
    
    results = predict_validation_set(model, imputer, scaler, X_val, val_df, train_cols)
    
    output_file = 'validation_predictions_adaboost.csv'
    results.to_csv(output_file, index=False)
    print(f"\n✅ Fichier sauvegardé: {output_file}")

if __name__ == "__main__":
    main()
