import pandas as pd
import numpy as np

# Charger le CSV
df = pd.read_csv("ton_fichier.csv")

# Colonnes représentant des temps vers des événements
event_cols = ['TIME_TO_PARADE_1', 'TIME_TO_PARADE_2', 'TIME_TO_NIGHT_SHOW']

# Remplacer les valeurs vides / NaN par np.inf pour les événements
for col in event_cols:
    df[col] = df[col].replace(0, np.nan)  # si des 0 ont déjà été mis
    df[col] = df[col].fillna(np.inf)

# Colonnes numériques restantes à remplir avec 0
numeric_cols = ['ADJUST_CAPACITY', 'DOWNTIME', 'CURRENT_WAIT_TIME']
df[numeric_cols] = df[numeric_cols].fillna(0)

# Convertir la colonne datetime
df['DATETIME'] = pd.to_datetime(df['DATETIME'])

# Features temporelles de base
df['year'] = df['DATETIME'].dt.year
df['month'] = df['DATETIME'].dt.month
df['day'] = df['DATETIME'].dt.day
df['hour'] = df['DATETIME'].dt.hour
df['minute'] = df['DATETIME'].dt.minute
df['second'] = df['DATETIME'].dt.second

# Nouvelle colonne : jour de la semaine (Lundi=0, Dimanche=6)
df['day_of_week'] = df['DATETIME'].dt.weekday

# Indicateur week-end
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

# One-hot encoding de la catégorie en supprimant la première colonne de référence
df = pd.get_dummies(df, columns=['ENTITY_DESCRIPTION_SHORT'], drop_first=True)

# Supprimer la colonne datetime originale
df = df.drop(columns=['DATETIME'])

# Vérification
print(df.head())
