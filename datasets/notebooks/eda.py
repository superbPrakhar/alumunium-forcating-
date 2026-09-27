import pandas as pd

al = pd.read_csv("datasets/Aluminium Historical Data_2012.csv")

# Convert Date
al["Date"] = pd.to_datetime(al["Date"])

# Remove commas from price columns
for col in ["Price","Open","High","Low"]:
    al[col] = al[col].str.replace(",","")
    al[col] = pd.to_numeric(al[col])

print(al.info())

print("\nSUMMARY")
print(al.describe())

print("\nFIRST 5 ROWS")
print(al.head())