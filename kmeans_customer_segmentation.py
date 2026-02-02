from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load the dataset
df = pd.read_csv('customer_data_5000.csv')

print("Dataset loaded successfully!")
print(f"Total records: {len(df)}")
print("\nFirst few rows:")
print(df.head())

# Select features for clustering
features = ['age', 'annual_income', 'spending_score']
X = df[features].copy()

print("\n" + "="*60)
print("FEATURE STATISTICS")
print("="*60)
print(X.describe())

# Standardize the features (important for K-means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using Elbow Method
print("\n" + "="*60)
print("FINDING OPTIMAL NUMBER OF CLUSTERS")
print("="*60)

inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    print(
        f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette Score={silhouette_scores[-1]:.3f}")

# Plot Elbow Method
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(K_range, silhouette_scores, 'ro-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Analysis')
plt.grid(True)

# Based on analysis, let's use K=5 (you can change this)
optimal_k = 5
print(f"\n{'='*60}")
print(f"APPLYING K-MEANS WITH K={optimal_k} CLUSTERS")
print("="*60)

# Apply K-means with optimal K
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Analyze clusters
print("\nCluster Distribution:")
print(df['cluster'].value_counts().sort_index())

print("\nCluster Characteristics:")
cluster_summary = df.groupby('cluster')[features].mean()
print(cluster_summary)

# Assign meaningful names to clusters based on characteristics


def assign_cluster_name(row):
    if row['annual_income'] > 100000 and row['spending_score'] > 60:
        return 'High-Value Spenders'
    elif row['annual_income'] > 100000 and row['spending_score'] <= 60:
        return 'High-Income Conservative'
    elif row['annual_income'] <= 60000 and row['spending_score'] > 60:
        return 'Budget Enthusiasts'
    elif row['annual_income'] <= 60000 and row['spending_score'] <= 60:
        return 'Cost-Conscious'
    else:
        return 'Middle-Income Moderate'


cluster_names = cluster_summary.apply(assign_cluster_name, axis=1)
print("\nCluster Names:")
for cluster_id, name in cluster_names.items():
    print(f"  Cluster {cluster_id}: {name}")

# Map cluster names to dataframe
df['cluster_name'] = df['cluster'].map(cluster_names.to_dict())

# Visualize clusters
plt.subplot(1, 3, 3)
scatter = plt.scatter(df['annual_income'], df['spending_score'],
                      c=df['cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('Annual Income ($)')
plt.ylabel('Spending Score')
plt.title(f'Customer Segments (K={optimal_k})')
plt.colorbar(scatter, label='Cluster')

plt.tight_layout()
plt.savefig('kmeans_analysis.png', dpi=300, bbox_inches='tight')
print("\nElbow and clustering plots saved as 'kmeans_analysis.png'")

# Create detailed visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Annual Income vs Spending Score
axes[0, 0].scatter(df['annual_income'], df['spending_score'],
                   c=df['cluster'], cmap='viridis', alpha=0.6, s=50)
axes[0, 0].set_xlabel('Annual Income ($)')
axes[0, 0].set_ylabel('Spending Score')
axes[0, 0].set_title('Income vs Spending Score by Cluster')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Age vs Spending Score
axes[0, 1].scatter(df['age'], df['spending_score'],
                   c=df['cluster'], cmap='viridis', alpha=0.6, s=50)
axes[0, 1].set_xlabel('Age')
axes[0, 1].set_ylabel('Spending Score')
axes[0, 1].set_title('Age vs Spending Score by Cluster')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Age vs Annual Income
axes[1, 0].scatter(df['age'], df['annual_income'],
                   c=df['cluster'], cmap='viridis', alpha=0.6, s=50)
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Annual Income ($)')
axes[1, 0].set_title('Age vs Income by Cluster')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Cluster size distribution
cluster_counts = df['cluster_name'].value_counts()
axes[1, 1].barh(range(len(cluster_counts)),
                cluster_counts.values, color='skyblue')
axes[1, 1].set_yticks(range(len(cluster_counts)))
axes[1, 1].set_yticklabels(cluster_counts.index)
axes[1, 1].set_xlabel('Number of Customers')
axes[1, 1].set_title('Customer Distribution by Segment')
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('customer_segments_detailed.png', dpi=300, bbox_inches='tight')
print("Detailed segmentation plots saved as 'customer_segments_detailed.png'")

# Create 3D visualization

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(df['age'], df['annual_income'], df['spending_score'],
                     c=df['cluster'], cmap='viridis', alpha=0.6, s=50)

ax.set_xlabel('Age')
ax.set_ylabel('Annual Income ($)')
ax.set_zlabel('Spending Score')
ax.set_title(f'3D Customer Segmentation (K={optimal_k} Clusters)')
plt.colorbar(scatter, label='Cluster', shrink=0.5)

plt.savefig('customer_segments_3d.png', dpi=300, bbox_inches='tight')
print("3D visualization saved as 'customer_segments_3d.png'")

# Save the clustered data
output_file = 'customer_data_clustered.csv'
df.to_csv(output_file, index=False)
print(f"\nClustered data saved to '{output_file}'")

# Print detailed cluster analysis
print("\n" + "="*60)
print("DETAILED CLUSTER ANALYSIS")
print("="*60)

for cluster_id in sorted(df['cluster'].unique()):
    cluster_data = df[df['cluster'] == cluster_id]
    cluster_name = cluster_names[cluster_id]

    print(f"\n{'='*60}")
    print(f"CLUSTER {cluster_id}: {cluster_name}")
    print(f"{'='*60}")
    print(
        f"Size: {len(cluster_data)} customers ({len(cluster_data)/len(df)*100:.1f}%)")
    print(f"\nAverage Characteristics:")
    print(f"  Age: {cluster_data['age'].mean():.1f} years")
    print(f"  Annual Income: ${cluster_data['annual_income'].mean():.2f}")
    print(f"  Spending Score: {cluster_data['spending_score'].mean():.1f}")
    print(
        f"\nAge Range: {cluster_data['age'].min()} - {cluster_data['age'].max()}")
    print(
        f"Income Range: ${cluster_data['annual_income'].min():.2f} - ${cluster_data['annual_income'].max():.2f}")
    print(
        f"Spending Range: {cluster_data['spending_score'].min()} - {cluster_data['spending_score'].max()}")

print("\n" + "="*60)
print("CLUSTERING COMPLETED SUCCESSFULLY!")
print("="*60)
plt.show()
