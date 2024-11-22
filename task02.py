import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df=pd.read_csv('Mall_Customers.csv')
print(df.head())

# Drop CustomerID and Gender for clustering without the need to  convert Gender to numeric 
df_numeric = df.drop(columns=["CustomerID", "Gender"])

# Standardize the features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)

# Set the number of clusters
k = 3  # we can try different values for k to see what works best

# Create and fit the model
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(df_scaled)
# Add cluster labels to the original dataframe
df["Cluster"] = kmeans.labels_

# Plot the clusters based on two key features (e.g., Annual Income and Spending Score)
plt.figure(figsize=(10, 6))
plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"], c=df["Cluster"], cmap='viridis', marker="o")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("K-means Clustering of Customers")
plt.colorbar(label="Cluster Label")
plt.show()
# Display the cluster centers
print("Cluster Centers:\n", scaler.inverse_transform(kmeans.cluster_centers_))