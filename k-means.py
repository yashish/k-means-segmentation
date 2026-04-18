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






