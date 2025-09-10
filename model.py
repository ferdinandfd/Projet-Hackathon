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

df_val_attrac_dt = df_val[["DATETIME", "ENTITY_DESCRIPTION_SHORT"]].copy() #à garder car on les veut à la fin


#Régler pb cases vides pour les temps d'attente d'évènements
event_cols = ['TIME_TO_PARADE_1', 'TIME_TO_PARADE_2', 'TIME_TO_NIGHT_SHOW']

for col in event_cols:
    df_train[col] = df_train[col].fillna(100000) #temps très long avant parade équivaut à pas de parade
    df_val[col] = df_val[col].fillna(100000)


#Features temporelles: conversion en cyclique + suppression des colonnes inutiles
def conversion_feature_tempo(df):
    df['year'] = df['DATETIME'].dt.year
    df['month'] = df['DATETIME'].dt.month
    df['day'] = df['DATETIME'].dt.day
    df['hour'] = df['DATETIME'].dt.hour
    df['minute'] = df['DATETIME'].dt.minute
    df['second'] = df['DATETIME'].dt.second
    df['day_of_week'] = df['DATETIME'].dt.weekday
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
    df['minute_sin'] = np.sin(2*np.pi*df['minute']/60)
    df['minute_cos'] = np.cos(2*np.pi*df['minute']/60)
    df['day_of_week_sin'] = np.sin(2*np.pi*df['day_of_week']/7)
    df['day_of_week_cos'] = np.cos(2*np.pi*df['day_of_week']/7)
    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)
    df = df.drop(columns=['month', 'day', 'hour', 'minute', 'second', 'DATETIME'])
    return df

df_train = conversion_feature_tempo(df_train)
df_val = conversion_feature_tempo(df_val)


#One-hot encoding des noms d'attraction
df_train = pd.get_dummies(df_train, columns=['ENTITY_DESCRIPTION_SHORT'], drop_first=True)
df_val = pd.get_dummies(df_val, columns=['ENTITY_DESCRIPTION_SHORT'], drop_first=True)

#Définir X_train, Y_train, X_val
features = df_train.columns.drop("WAIT_TIME_IN_2H")
X_train = df_train[features]
Y_train = df_train['WAIT_TIME_IN_2H']
X_val = df_val


##MODELE
model = LinearRegression()
model.fit(X_train, Y_train)

Y_val_pred = model.predict(X_val)

#Rendu
df_result = df_val_attrac_dt.copy()
df_result["y_pred"] = Y_val_pred
df_result["KEY"] = "Validation"

df_result.to_csv("C:/Users/Utilisateur/Documents/MyGitDirectory/Projet-Hackathon/y_pred_linear.csv", index=False, encoding="utf-8")