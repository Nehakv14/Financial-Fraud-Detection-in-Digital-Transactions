import pandas as pd
import os

# Load transaction and identity data
transaction = pd.read_csv('data/train_transaction.csv')
identity = pd.read_csv('data/train_identity.csv')

print(f"Transaction shape: {transaction.shape}")
print(f"Identity shape: {identity.shape}")

# Merge on TransactionID
full_data = transaction.merge(identity, on='TransactionID', how='left')

print(f"Merged shape: {full_data.shape}")

# Save merged CSV
full_data.to_csv('data/train_full.csv', index=False)
print("Saved to data/train_full.csv")