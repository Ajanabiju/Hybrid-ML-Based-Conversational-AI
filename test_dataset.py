import pandas as pd

df = pd.read_csv("dataset.csv")

print(df.head())
print("Total rows:", len(df))
print("Tags:", df["tag"].unique())
