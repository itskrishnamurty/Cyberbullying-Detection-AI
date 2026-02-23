import pandas as pd

df = pd.read_csv("data/raw/cyberbullying_tweets.csv")

print("\nFirst 5 rows:\n")
print(df.head())

print("\nColumns:\n")
print(df.columns)

print("\nClass distribution:\n")
print(df.iloc[:, -1].value_counts())