import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv(
    "outputs/master_dataset_v2.csv"
)

# =========================
# USE ONLY MACRO FEATURES
# (NO LAG FEATURES)
# =========================

X = df[
    [
        "Oil_Price",
        "USD_Index",
        "Interest_Rate",
        "Inflation",
        "GDP",
        "Unemployment",
        "Ind_Prod",
        "sp500"
    ]
]

y = df["Price"]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    shuffle=False
)

# =========================
# MODELS
# =========================

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),

    "SVR": SVR()
}

# =========================
# TRAIN & EVALUATE
# =========================

for name, model in models.items():

    print("\n")
    print("=" * 40)
    print(name)
    print("=" * 40)

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = (
        mean_squared_error(
            y_test,
            predictions
        )
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2  :", round(r2, 4))

print("\nExperiment Complete")
print("This model uses ONLY macroeconomic variables.")