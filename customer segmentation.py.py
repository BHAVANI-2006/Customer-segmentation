import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("customers.csv")

X = df[['AnnualIncome', 'SpendingScore']]

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

print("Customer Segments:")
print(df[['CustomerID', 'AnnualIncome', 'SpendingScore', 'Cluster']])

print("\nCluster Centers:")
print(kmeans.cluster_centers_)

plt.figure(figsize=(8,6))
plt.scatter(
    X['AnnualIncome'],
    X['SpendingScore'],
    c=df['Cluster'],
    cmap='viridis',
    s=100
)

plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    marker='X',
    s=300,
    color='red',
    label='Centroids'
)

plt.title('Customer Segmentation')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()
