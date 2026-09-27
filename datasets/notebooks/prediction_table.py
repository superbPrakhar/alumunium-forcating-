import pandas as pd

from sklearn.linear_model import LinearRegression

df = pd.read_csv("outputs/master_dataset.csv")

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

y = df["Price"]

split = int(len(df)*0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

model = LinearRegression()
model.fit(X_train,y_train)

pred = model.predict(X_test)

results = pd.DataFrame({
    "Actual_Price": y_test,
    "Predicted_Price": pred
})

results["Difference"] = (
    results["Actual_Price"]
    - results["Predicted_Price"]
).abs()

results.to_csv(
    "outputs/predictions.csv",
    index=False
)

print(results.head())
print("\nSaved to outputs/predictions.csv")