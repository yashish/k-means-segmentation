import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_excel('Online Retail.xlsx', parse_dates=['InvoiceDate'])

print(df.shape)
print(df.head())
print(df.isnull().sum())

# Drop rows with missing CustomerID
df = df.dropna(subset=['CustomerID'])

df['CustomerID'] = df['CustomerID'].astype(int)

# Remove rows with missing InvoiceNo
# df = df[df['InvoiceNo'].notna()]

# This did not work as expected because some InvoiceNo are NaN, which is a float type, and the str.startswith method does not work on NaN values. Instead, we can use the str.startswith method with na=False to ignore NaN values.
# na=false will treat NaN values as False, so they will not be included in the filtered DataFrame. This way, we can remove canceled transactions without affecting rows with missing InvoiceNo. 
# and that worked as expected, we were able to remove canceled transactions while keeping rows with missing InvoiceNo intact.

# Remove canceled transactions
df = df[~df['InvoiceNo'].str.startswith('C', na=False)]

# Remove transactions with non-positive Quantity or UnitPrice (missing values/data errors)
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Create a revenue column
df['Revenue'] = df['Quantity'] * df['UnitPrice']

df.shape

# set reference date for recency calculation (day after the last transaction date)
reference_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',  # Frequency
    'Revenue': 'sum'  # Monetary
}).reset_index()

print(rfm.head())
print(rfm.describe())




# Create RFM features




