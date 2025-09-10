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


#Features temporelles: détail
def conversion_feature_tempo(df):
    df['year'] = df['DATETIME'].dt.year
    df['month'] = df['DATETIME'].dt.month
    df['day'] = df['DATETIME'].dt.day
    df['hour'] = df['DATETIME'].dt.hour
    df['minute'] = df['DATETIME'].dt.minute
    df['second'] = df['DATETIME'].dt.second
    df['day_of_week'] = df['DATETIME'].dt.weekday
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    return df

df_train = conversion_feature_tempo(df_train)
df_val = conversion_feature_tempo(df_val)


#One-hot encoding des noms d'attraction
df_train = pd.get_dummies(df_train, columns=['ENTITY_DESCRIPTION_SHORT'], drop_first=True)
df_val = pd.get_dummies(df_val, columns=['ENTITY_DESCRIPTION_SHORT'], drop_first=True)

""""
#Définir X_train, Y_train, X_val pour chaque attraction
df_train_wr = df_train[df_train["ENTITY_DESCRIPTION_SHORT_Water Ride"]==1] #water ride
df_train_ps = df_train[df_train["ENTITY_DESCRIPTION_SHORT_Pirate Ship"]==1] #pirate ship
df_train_fc = df_train[(df_train['ENTITY_DESCRIPTION_SHORT_Pirate Ship']==0) & (df_train['ENTITY_DESCRIPTION_SHORT_Water Ride']==0)] #flying coaster

X_val_wr = df_val[df_val["ENTITY_DESCRIPTION_SHORT_Water Ride"]==1] #water ride
X_val_ps = df_val[df_val['ENTITY_DESCRIPTION_SHORT_Pirate Ship']==1] #pirate ship
X_val_fc = df_val[(df_val['ENTITY_DESCRIPTION_SHORT_Pirate Ship']==0) & (df_val['ENTITY_DESCRIPTION_SHORT_Water Ride']==0)]#flying coaster

Y_train_wr = df_train_wr['WAIT_TIME_IN_2H']
X_train_wr = df_train_wr.drop("WAIT_TIME_IN_2H")

Y_train_ps = df_train_ps['WAIT_TIME_IN_2H']
X_train_ps = df_train_ps.drop("WAIT_TIME_IN_2H")

Y_train_fc = df_train_fc['WAIT_TIME_IN_2H']
X_train_fc = df_train_fc.drop("WAIT_TIME_IN_2H")

"""
"""
##MODELES
model_wr = LinearRegression()
model_ps = LinearRegression()
model_fc = LinearRegression()

model_wr.fit(X_train_wr, Y_train_wr)
model_ps.fit(X_train_ps, Y_train_ps)
model_fc.fit(X_train_fc, Y_train_fc)

Y_pred_wr = model_wr.predict(X_val_wr)
Y_pred_ps = model_ps.predict(X_val_ps)
Y_pred_fc = model_fc.predict(X_val_fc)
"""

#Rendu
"""
df_result = df_val_attrac_dt.copy()
df_ypred = pd.concat(Y_pred_wr, Y_pred_ps, Y_pred_fc)
df_ypred = df_ypred.sort_values('index')  # remet dans l'ordre original
df_result["y_pred"] = df_ypred
df_result["KEY"] = "Validation"
"""
df_val.to_csv("C:/Users/Utilisateur/Documents/MyGitDirectory/Projet-Hackathon/xval_preproc_2.csv", index=False, encoding="utf-8")
df_train.to_csv("C:/Users/Utilisateur/Documents/MyGitDirectory/Projet-Hackathon/xtrain_preproc_2.csv", index=False, encoding="utf-8")
#df_result.to_csv("C:/Users/Utilisateur/Documents/MyGitDirectory/Projet-Hackathon/y_linear_par_attract.csv", index=False, encoding="utf-8")