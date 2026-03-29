import pandas as pd
import numpy as np
import pickle
from keras.models import load_model

# Load model and scaler
model = load_model("models/lstm_model.h5")

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load clean data
df = pd.read_csv("models/clean_lstm_data.csv")
values = df[['value']].values

# Use last 60 days for prediction
last_60 = values[-60:]

# Scale input
scaled_input = scaler.transform(last_60)
X_input = scaled_input.reshape(1, 60, 1)

# Predict
prediction = model.predict(X_input)

# Inverse scale
predicted_price = scaler.inverse_transform(prediction)

print("Predicted next price:", predicted_price[0][0])
