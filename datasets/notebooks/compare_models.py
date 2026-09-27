import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

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

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    "SVR": SVR()
}

for name, model in models.items():

    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test,pred)

    rmse = mean_squared_error(
        y_test,
        pred
    )**0.5

    r2 = r2_score(
        y_test,
        pred
    )

    print("\n",name)
    print("MAE :",round(mae,2))
    print("RMSE:",round(rmse,2))
    print("R2  :",round(r2,4))