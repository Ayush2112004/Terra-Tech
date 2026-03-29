import pandas as pd
df = pd.read_csv("models/data.csv")

df = df[['date', 'value']]
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

df.to_csv("models/clean_lstm_data.csv", index=False)
