import pandas as pd

df = pd.read_csv("data/processed/cleaned_data.csv")

print("\nColumns:\n", df.columns)

print("\nSample data:\n")
print(df.head())
#-------------------------------#
print("\nUnique labels:\n")
print(df['cyberbullying_type'].unique())

#-------------------------------#
df['label'] = df['cyberbullying_type'].apply(
    lambda x: 0 if x == 'not_cyberbullying' else 1
)

print("\nLabel counts:\n")
print(df['label'].value_counts())
#-------------------------------#
import matplotlib.pyplot as plt

df['label'].value_counts().plot(kind='bar')
plt.title("Bullying vs Normal Posts")
plt.xticks([0,1], ["Normal","Bullying"], rotation=0)
plt.show()
#-------------------------------#
df.to_csv("data/processed/final_data.csv", index=False)