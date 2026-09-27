import pandas as pd

# =========================
# LOAD DATASETS
# =========================

al = pd.read_csv("datasets/Aluminium Historical Data_2012.csv")

oil = pd.read_csv("datasets/wti_full.csv")

usd = pd.read_csv("datasets/US Dollar Index Historical Data.csv")

macro = pd.read_csv("datasets/main.csv")

# =========================
# CLEAN ALUMINIUM
# =========================

al["Date"] = pd.to_datetime(al["Date"])

al["Price"] = al["Price"].astype(str).str.replace(",", "")
al["Price"] = pd.to_numeric(al["Price"])

al = al[["Date", "Price"]]

# =========================
# CLEAN OIL
# =========================

oil.rename(
    columns={
        "DATE": "Date",
        "DCOILWTICO": "Oil_Price"
    },
    inplace=True
)

oil["Date"] = pd.to_datetime(oil["Date"])

oil["Oil_Price"] = oil["Oil_Price"].ffill()

oil = oil[["Date", "Oil_Price"]]

# =========================
# CLEAN USD INDEX
# =========================

usd["Date"] = pd.to_datetime(usd["Date"])

usd["Price"] = usd["Price"].astype(str).str.replace(",", "")
usd["Price"] = pd.to_numeric(usd["Price"])

usd.rename(
    columns={
        "Price": "USD_Index"
    },
    inplace=True
)

usd = usd[["Date", "USD_Index"]]

# =========================
# CLEAN MACRO DATA
# =========================

macro.rename(
    columns={
        "date": "Date"
    },
    inplace=True
)

macro["Date"] = pd.to_datetime(macro["Date"])

# =========================
# SORT DATA
# =========================

al = al.sort_values("Date")
oil = oil.sort_values("Date")
usd = usd.sort_values("Date")
macro = macro.sort_values("Date")

# =========================
# MERGE DATASETS
# =========================

master = pd.merge(
    al,
    oil,
    on="Date",
    how="left"
)

master = pd.merge(
    master,
    usd,
    on="Date",
    how="left"
)

master = pd.merge(
    master,
    macro,
    on="Date",
    how="left"
)

# =========================
# FILL MISSING VALUES
# =========================

master = master.ffill()
master = master.bfill()

# =========================
# CREATE FEATURES
# =========================

master["Price_Lag_1"] = master["Price"].shift(1)

master["Price_Lag_7"] = master["Price"].shift(7)

master["Price_Lag_30"] = master["Price"].shift(30)

master["MA_7"] = master["Price"].rolling(7).mean()

master["MA_30"] = master["Price"].rolling(30).mean()

# =========================
# REMOVE NULL ROWS
# =========================

master = master.dropna()

# =========================
# SAVE DATASET
# =========================

master.to_csv(
    "outputs/master_dataset_v2.csv",
    index=False
)

print("\nMASTER DATASET V2 CREATED SUCCESSFULLY")

print("\nShape:")
print(master.shape)

print("\nColumns:")
print(master.columns)

print("\nFirst 5 Rows:")
print(master.head())