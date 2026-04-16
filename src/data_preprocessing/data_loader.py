from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

print("Top-level contents:")
for item in DATA_DIR.iterdir():
    print("-", item)

csv_files = list(DATA_DIR.rglob("*.csv"))
txt_files = list(DATA_DIR.rglob("*.txt"))

print("\nCSV files found:")
for f in csv_files:
    print("-", f)

print("\nTXT files found:")
for f in txt_files[:20]:
    print("-", f)

if csv_files:
    df = pd.read_csv(csv_files[0])
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumns:")
    print(df.columns.tolist())
else:
    print("\nNo CSV files found.")