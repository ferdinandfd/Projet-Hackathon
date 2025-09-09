import pandas as pd
import numpy as np

# Charger le CSV
df = pd.read_csv("ton_fichier.csv")

# Remplacer toutes les valeurs vides / NaN par 0
df = df.fillna(0)

# Convertir la colonne datetime
df['DATETIME'] = pd.to_datetime(df['DATETIME'])

# Features temporelles
df['year'] = df['DATETIME'].dt.year
df['month'] = df['DATETIME'].dt.month
df['day'] = df['DATETIME'].dt.day
df['hour'] = df['DATETIME'].dt.hour
df['minute'] = df['DATETIME'].dt.minute
df['weekday'] = df['DATETIME'].dt.weekday
df['is_weekend'] = df['weekday'].isin([5,6]).astype(int)

# Features cycliques
df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
df['weekday_sin'] = np.sin(2*np.pi*df['weekday']/7)
df['weekday_cos'] = np.cos(2*np.pi*df['weekday']/7)

# Encoder la catégorie
df = pd.get_dummies(df, columns=['ENTITY_DESCRIPTION_SHORT'])

# Supprimer la colonne datetime si nécessaire
df = df.drop(columns=['DATETIME'])

# Vérification
print(df.head())
