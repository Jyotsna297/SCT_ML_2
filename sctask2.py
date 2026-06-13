import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load Dataset
df = pd.read_csv("Mall_Customers.csv")

# Display Dataset Information
print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

# Select Features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# -------------------------------
# Elbow Method
# -------------------------------
wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42,
        n_init=10
    )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# -------------------------------
# K-Means Clustering
# -------------------------------
kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42,
    n_init=10
)

y_kmeans = kmeans.fit_predict(X)

# Add Cluster Labels
df['Cluster'] = y_kmeans

print("\nClustered Dataset:")
print(df.head())

# -------------------------------
# Customer Segmentation Plot
# -------------------------------
plt.figure(figsize=(10, 6))

plt.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=y_kmeans,
    cmap='viridis',
    s=80
)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    c='red',
    marker='X',
    label='Centroids'
)

plt.title('Customer Segmentation using K-Means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)

plt.show()

# Save Result
df.to_csv("Mall_Customers_Clustered.csv", index=False)

print("\nClustered dataset saved as 'Mall_Customers_Clustered.csv'")
print("\nCustomers in each cluster:")
print(df['Cluster'].value_counts())