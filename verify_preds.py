import pandas as pd
from torchmetrics.regression import PearsonCorrCoef
import torch
r = PearsonCorrCoef(num_outputs=6)
#        r = r(preds, labels)
#        r = r.mean()


# Load the CSV file into a DataFrame
df_preds = pd.read_csv('preds_modified.csv')
df_val = pd.read_csv('data/valid_split.csv')
print(df_preds.head())
print(df_val.head())
print(df_preds.values.shape)
pearson = r(torch.tensor(df_preds.values[:,1:]), torch.tensor(df_val.values[:,1:]))
print(pearson.mean())
# Columns to check for negative values
columns_to_check = ['Admiration', 'Amusement', 'Determination', 'Empathic Pain', 'Excitement', 'Joy']

# Set negative values to 0.0 in the specified columns
for column in columns_to_check:
    df_preds[column] = df_preds[column].apply(lambda x: max(x, 0.0)) 
    df_preds[column] = df_preds[column].apply(lambda x: min(x, 1.0)) 
# Save the modified DataFrame back to 'pred.csv'

pearson = r(torch.tensor(df_preds.values[:,1:]), torch.tensor(df_val.values[:,1:]))
print(pearson.mean())
