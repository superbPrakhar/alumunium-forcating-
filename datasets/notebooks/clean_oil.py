import pandas as pd

oil = pd.read_csv("datasets/wti_full.csv")

print("\nFIRST 5 ROWS")
print(oil.head())

print("\nINFO")
print(oil.info())

print("\nMISSING VALUES")
print(oil.isnull().sum())