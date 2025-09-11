import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


##PREPARATION DES DONNEES
#charger CSV
df_main = pd.read_csv("waiting_times_train.csv")  #horaires
df_meteo = pd.read_csv("weather_data.csv") #meteo
X_val_main = pd.read_csv("waiting_times_X_test_val.csv") #validation


#supprimer colonnes considérées inutiles ds meteo (par étude du tableau "à la main")
df_meteo = df_meteo.drop(columns=['temp', 'dew_point', 'pressure','snow_1h'])


#conversion en datetime pour pouvoir merge
df_main["DATETIME"] = pd.to_datetime(df_main["DATETIME"])
df_meteo["DATETIME"] = pd.to_datetime(df_meteo["DATETIME"])
X_val_main["DATETIME"] = pd.to_datetime(X_val_main["DATETIME"])


#Fusion avec la météo pour le train set et le validation set 
df_train = pd.merge(
    df_main,
    df_meteo,
    on = "DATETIME",
    how = "left" #si info meteo pour horaire non présent dans waiting_times_train, on supp la ligne de meteo
)

df_val = pd.merge(
    X_val_main,
    df_meteo,
    on = "DATETIME",
    how = "left"
)


#Régler pb cases vides pour les temps d'attente d'évènements
event_cols = ['TIME_TO_PARADE_1', 'TIME_TO_PARADE_2', 'TIME_TO_NIGHT_SHOW']

for col in event_cols:
    df_train[col] = df_train[col].fillna(100000) #temps très long avant parade équivaut à pas de parade
    df_val[col] = df_val[col].fillna(100000)


df_train['year'] = df_train['DATETIME'].dt.year
df_train['month'] = df_train['DATETIME'].dt.month
df_train['day'] = df_train['DATETIME'].dt.day
df_train['hour'] = df_train['DATETIME'].dt.hour
df_train['minute'] = df_train['DATETIME'].dt.minute
df_train['second'] = df_train['DATETIME'].dt.second
df_train['day_of_week'] = df_train['DATETIME'].dt.weekday
df_train['is_weekend'] = df_train['day_of_week'].isin([5,6]).astype(int)

# Pour df_val
df_val['year'] = df_val['DATETIME'].dt.year
df_val['month'] = df_val['DATETIME'].dt.month
df_val['day'] = df_val['DATETIME'].dt.day
df_val['hour'] = df_val['DATETIME'].dt.hour
df_val['minute'] = df_val['DATETIME'].dt.minute
df_val['second'] = df_val['DATETIME'].dt.second
df_val['day_of_week'] = df_val['DATETIME'].dt.weekday
df_val['is_weekend'] = df_val['day_of_week'].isin([5,6]).astype(int)


#One-hot encoding des noms d'attraction
df_train = pd.get_dummies(df_train, columns=['ENTITY_DESCRIPTION_SHORT'], drop_first=True)
df_val = pd.get_dummies(df_val, columns=['ENTITY_DESCRIPTION_SHORT'], drop_first=True)



##GRAPHIQUES----------------------------------------------------------------------------------

import matplotlib.pyplot as plt

# Répartition des années dans df_train
year_counts_train = df_train['year'].value_counts().sort_index()
# Répartition des années dans df_val
year_counts_val = df_val['year'].value_counts().sort_index()


##GRAPHIQUES CAMEMBERT
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

#Graphique camembert années train
axes[0].pie(year_counts_train, labels=year_counts_train.index, autopct='%1.1f%%')
axes[0].set_title("Répartition des années - Train")

#Graphique camembert années val
axes[1].pie(year_counts_val, labels=year_counts_val.index, autopct='%1.1f%%')
axes[1].set_title("Répartition des années - Validation")

plt.show()


##HISTOGRAMME JOURS SEMAINE
dow_counts_train = df_train['day_of_week'].value_counts().sort_index()
dow_counts_val = df_val['day_of_week'].value_counts().sort_index()

#Jours
jours = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']

# Histogramme côte à côte
x = np.arange(len(jours))  # positions des barres
width = 0.35               # largeur des barres

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, dow_counts_train, width, label='Train')
ax.bar(x + width/2, dow_counts_val, width, label='Validation')

# Mise en forme
ax.set_xticks(x)
ax.set_xticklabels(jours)
ax.set_ylabel("Nombre d'observations")
ax.set_title("Répartition des jours de la semaine")
ax.legend()

plt.show()