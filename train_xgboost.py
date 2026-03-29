import pandas as pd
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


df = pd.read_csv("models/data.csv")


df["date"] = pd.to_datetime(df["date"])

df["dayofweek"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["dayofyear"] = df["date"].dt.dayofyear

df["lag_1"] = df["value"].shift(1)
df["lag_2"] = df["value"].shift(2)
df["lag_3"] = df["value"].shift(3)

df["rolling_mean_3"] = df["value"].rolling(3).mean()
df["rolling_std_3"] = df["value"].rolling(3).std()

df = df.dropna()


X = df.drop(columns=["date", "value", "scaled_value"], errors="ignore")
y = df["value"]

print("XGBoost feature columns:", X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)



model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)


preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print("XGBoost MAE:", mae)


with open("models/xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("XGBoost model saved successfully!")
