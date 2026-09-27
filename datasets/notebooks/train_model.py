import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# LOAD DATA

df = pd.read_csv("outputs/master_dataset.csv")

# FEATURES

X = df[
    [
        "Oil_Price",
        "USD_Index",
        "Price_Lag_1",
        "Price_Lag_7",
        "Price_Lag_30",
        "MA_7",
        "MA_30"
    ]
]

# TARGET

y = df["Price"]

# TRAIN TEST SPLIT

split_index = int(len(df) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

# MODEL

model = LinearRegression()

model.fit(X_train, y_train)

# PREDICT

predictions = model.predict(X_test)

# METRICS

mae = mean_absolute_error(y_test, predictions)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)

print("\nRESULTS")
print("MAE :", round(mae,2))
print("RMSE:", round(rmse,2))
print("R2  :", round(r2,4))