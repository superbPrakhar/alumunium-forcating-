import pandas as pd

# Load datasets

al = pd.read_csv("datasets/Aluminium Historical Data_2012.csv")
oil = pd.read_csv("datasets/wti_full.csv")
usd = pd.read_csv("datasets/US Dollar Index Historical Data.csv")
macro = pd.read_csv("datasets/main.csv")

print("\n========== ALUMINIUM ==========")
print(al.head())
print(al.shape)

print("\n========== OIL ==========")
print(oil.head())
print(oil.shape)

print("\n========== USD ==========")
print(usd.head())
print(usd.shape)

print("\n========== MACRO ==========")
print(macro.head())
print(macro.shape)