import pandas as pd
import numpy as np
import os

def preprocess_csv(file_path):
    """
    Transforme un CSV :
    - Remplace NaN par np.inf pour les colonnes d'événements
    - Remplace NaN par 0 pour les autres colonnes numériques
    - Génère les features temporelles et cycliques
    - One-hot encode ENTITY_DESCRIPTION_SHORT
    - Ajoute les colonnes binaires is_there_*
    - Supprime DATETIME
    - Sauvegarde dans le même dossier avec '_preprocessed' ajouté au nom
    """
    
    df = pd.read_csv(file_path)
    
    # Colonnes représentant des temps vers des événements
    event_cols = ['TIME_TO_PARADE_1', 'TIME_TO_PARADE_2', 'TIME_TO_NIGHT_SHOW']
    
    for col in event_cols:
        df[col] = df[col].fillna(100000)
    
    # Colonnes numériques restantes à remplir avec 0
    numeric_cols = ['ADJUST_CAPACITY', 'DOWNTIME', 'CURRENT_WAIT_TIME']
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Convertir datetime
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    
    # Features temporelles de base
    df['year'] = df['DATETIME'].dt.year
    df['month'] = df['DATETIME'].dt.month
    df['day'] = df['DATETIME'].dt.day
    df['hour'] = df['DATETIME'].dt.hour
    df['minute'] = df['DATETIME'].dt.minute
    df['second'] = df['DATETIME'].dt.second
    df['day_of_week'] = df['DATETIME'].dt.weekday
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    
    # Features cycliques
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
    df['minute_sin'] = np.sin(2*np.pi*df['minute']/60)
    df['minute_cos'] = np.cos(2*np.pi*df['minute']/60)
    df['day_of_week_sin'] = np.sin(2*np.pi*df['day_of_week']/7)
    df['day_of_week_cos'] = np.cos(2*np.pi*df['day_of_week']/7)
    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)
    
    # One-hot encoding
    df = pd.get_dummies(df, columns=['ENTITY_DESCRIPTION_SHORT'], drop_first=True)
    
    # Ajouter colonnes binaires is_there_*
    df['is_there_parade_1'] = np.where(df['TIME_TO_PARADE_1'] != np.inf, 1, 0)
    df['is_there_parade_2'] = np.where(df['TIME_TO_PARADE_2'] != np.inf, 1, 0)
    df['is_there_night_show'] = np.where(df['TIME_TO_NIGHT_SHOW'] != np.inf, 1, 0)
    
    # Supprimer DATETIME
    df = df.drop(columns=['DATETIME'])
    
    # Sauvegarder le CSV transformé
    folder = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)
    new_file_name = f"{name}_preprocessed{ext}"
    save_path = os.path.join(folder, new_file_name)
    
    df.to_csv(save_path, index=False)
    
    return df, save_path

# -------------------------------
# Transformer tes 2 fichiers
# -------------------------------
df_train, path_train = preprocess_csv("waiting_times_train.csv")
df_test, path_test = preprocess_csv("waiting_times_X_test_val.csv")


print("Fichiers transformés sauvegardés ici :")
print(path_train)
print(path_test)
