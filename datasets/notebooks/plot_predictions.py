import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/predictions.csv")

plt.figure(figsize=(12,6))

plt.plot(
    df["Actual_Price"].values,
    label="Actual Price"
)

plt.plot(
    df["Predicted_Price"].values,
    label="Predicted Price"
)

plt.title("Actual vs Predicted Aluminium Price")

plt.xlabel("Test Data Points")
plt.ylabel("Price")

plt.legend()

plt.grid(True)

plt.savefig(
    "outputs/actual_vs_predicted.png"
)

plt.show()