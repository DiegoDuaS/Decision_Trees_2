import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

def breif_clustering(X, n_clusters):

    X_pca = PCA(n_components=2).fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42).fit(X_pca)

    X['Cluster'] = km.fit_predict(X_pca)
    centroides = km.cluster_centers_
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=X['Cluster'], palette="tab10", legend="full")
    # Plot centroids
    plt.scatter(centroides[:, 0], centroides[:, 1], c='red', marker='X', s=200, label="Centroids")

    plt.title(f"K-Means Clustering (Reducido con PCA) - 3 Clusters")
    plt.legend()
    plt.show()
    return X