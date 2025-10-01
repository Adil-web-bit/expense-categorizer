import pandas as pd

# yahan correct file ka naam likho
df = pd.read_csv("personal_expense_classification.csv")

print("---- First 5 rows ----")
print(df.head())

print("\n---- Columns ----")
print(df.columns)

print("\n---- Shape (rows, cols) ----")
print(df.shape)

print("\n---- Category distribution ----")
print(df['category'].value_counts())
