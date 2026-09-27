import pandas as pd

usd = pd.read_csv("datasets/US Dollar Index Historical Data.csv")

print("\nFIRST 5 ROWS")
print(usd.head())

print("\nINFO")
print(usd.info())

print("\nMISSING VALUES")
print(usd.isnull().sum())