import pandas as pd

df1 = pd.read_csv('waiting_times_train.csv')
df2 = pd.read_csv('weather_data.csv')

dfm = pd.merge(df1, df2)

print(df1.shape)
print(df2.shape)
print(dfm.shape)
