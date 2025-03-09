import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

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

def metrics_and_cm(y_pred, y_test):
    # Presicion
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    
    # Matriz de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def cv_metrics_and_cm(cv_scores, y, y_pred):
    # PARA MODELO DE VALIDACION CRUZADA
    # Presicion
    print("Precision promedio del modelo:", cv_scores.mean())
    print(classification_report(y, y_pred))
    cm = confusion_matrix(y, y_pred)

    # Matriz de confusion
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()