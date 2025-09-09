import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# 1️⃣ Charger et préparer le CSV
# -------------------------------
df = pd.read_csv("train.csv")  # ton CSV brut

# Colonnes représentant des temps vers des événements
event_cols = ['TIME_TO_PARADE_1', 'TIME_TO_PARADE_2', 'TIME_TO_NIGHT_SHOW']

# Remplacer les valeurs vides par np.inf pour les événements
for col in event_cols:
    df[col] = df[col].replace(0, np.nan)  # si 0 déjà présent
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

# One-hot encoding des catégories
df = pd.get_dummies(df, columns=['ENTITY_DESCRIPTION_SHORT'], drop_first=True)

# Supprimer DATETIME
df = df.drop(columns=['DATETIME'])

# -------------------------------
# 2️⃣ Séparer features globales et locales
# -------------------------------
global_features = ['year', 'month', 'day_of_week', 'is_weekend',
                   'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos']

local_features = ['hour', 'minute', 'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
                  'ADJUST_CAPACITY', 'DOWNTIME'] + event_cols

# Ajouter colonnes one-hot comme locales
local_features += [col for col in df.columns if col.startswith('ENTITY_DESCRIPTION_SHORT_')]

# 3️⃣ Définir X et y
X_global = df[global_features]
X_local = df[local_features]
y = df['CURRENT_WAIT_TIME']

# 4️⃣ Train/test split
Xg_train, Xg_test, Xl_train, Xl_test, y_train, y_test = train_test_split(
    X_global, X_local, y, test_size=0.2, random_state=42
)

# -------------------------------
# 5️⃣ Modèle global
# -------------------------------
model_global = LinearRegression()
model_global.fit(Xg_train, y_train)

y_global_train = model_global.predict(Xg_train)
y_global_test = model_global.predict(Xg_test)

# -------------------------------
# 6️⃣ Modèle local combiné
# -------------------------------
Xl_train_combined = Xl_train.copy()
Xl_train_combined['pred_global'] = y_global_train

Xl_test_combined = Xl_test.copy()
Xl_test_combined['pred_global'] = y_global_test

model_local = LinearRegression()
model_local.fit(Xl_train_combined, y_train)

# -------------------------------
# 7️⃣ Prédictions finales
# -------------------------------
y_pred = model_local.predict(Xl_test_combined)

# -------------------------------
# 8️⃣ Évaluation
# -------------------------------
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MSE :", mse)
print("RMSE :", rmse)
print("R2 :", r2)
