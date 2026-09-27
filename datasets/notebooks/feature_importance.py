import pandas as pd
from sklearn.linear_model import LinearRegression

# LOAD DATA

df = pd.read_csv(
    "outputs/master_dataset_v2.csv"
)

# FEATURES

X = df.drop(
    columns=["Date", "Price"]
)

y = df["Price"]

# TRAIN MODEL

model = LinearRegression()

model.fit(X, y)

# FEATURE IMPORTANCE

importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

importance["Absolute_Value"] = (
    importance["Coefficient"].abs()
)

importance = importance.sort_values(
    "Absolute_Value",
    ascending=False
)

print("\nFEATURE IMPORTANCE\n")

print(
    importance[
        ["Feature", "Coefficient"]
    ]
)