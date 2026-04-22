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

# Create RFM features
rfm = df.groupby('CustomerID').agg(
    Recency = ('InvoiceDate', lambda x: (reference_date - x.max()).days),  # Recency
    Frequency = ('InvoiceNo', 'nunique'),  # Frequency
    Monetary = ('Revenue', 'sum')  # Monetary
).reset_index()

print(rfm.head())
print(rfm.describe())

# Handle outliers and scale
rfm_log = rfm[['Recency','Frequency','Monetary']].copy()
rfm_log['Recency'] = np.log1p(rfm_log['Recency'])
rfm_log['Frequency'] = np.log1p(rfm_log['Frequency'])
rfm_log['Monetary'] = np.log1p(rfm_log['Monetary'])

# Scale so all features are on the same range
scalar = StandardScaler()
rfm_scaled = scalar.fit_transform(rfm_log)

print(rfm_scaled[:5])  # Print the first 5 rows of the scaled RFM features

# Determine optimal number of clusters using the elbow method
inertia = [] # List to store inertia values for different k values
k_range = range(1, 11) # We will test k values from 1 to 10 to find the optimal number of clusters using the elbow method.

# We will fit a KMeans model for each k value in the specified range and calculate the inertia 
# (within-cluster sum of squares) for each model. The inertia values will be stored in the list, 
# which we will then plot to visualize the elbow curve and determine the optimal number of clusters.
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # n_init=10 is the default value in scikit-learn 1.4 and later, but we specify it here for clarity and to ensure consistent results across different versions of scikit-learn.
    kmeans.fit(rfm_scaled) # Fit the KMeans model to the scaled RFM data
    inertia.append(kmeans.inertia_) # Append the inertia (within-cluster sum of squares) to the list for each k

# Plot the elbow curve
plt.figure(figsize=(8, 5)) # Set the figure size for better visibility
plt.plot(k_range, inertia, marker='o', linestyle='-', color='steelblue') # Plot the inertia values against the k values with markers and a line
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method - how many cluster segments?')
plt.tight_layout() # Adjust the layout to prevent overlap of labels and title
plt.show()

# From the elbow plot, we can choose k=4 as the optimal number of clusters

