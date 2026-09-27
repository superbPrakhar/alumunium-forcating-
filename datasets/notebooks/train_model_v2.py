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

# =====================
# LOAD DATA
# =====================

df = pd.read_csv(
    "outputs/master_dataset_v2.csv"
)

# =====================
# FEATURES
# =====================

X = df.drop(
    columns=["Date", "Price"]
)

y = df["Price"]

# =====================
# TRAIN TEST SPLIT
# =====================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# =====================
# MODELS
# =====================

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),
    "SVR": SVR()
}

# =====================
# TRAIN + EVALUATE
# =====================

for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        pred
    )

    rmse = (
        mean_squared_error(
            y_test,
            pred
        )
    ) ** 0.5

    r2 = r2_score(
        y_test,
        pred
    )

    print("\n" + "="*40)
    print(name)
    print("="*40)

    print("MAE :", round(mae,2))
    print("RMSE:", round(rmse,2))
    print("R2  :", round(r2,4))