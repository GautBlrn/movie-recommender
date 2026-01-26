from sklearn.neighbors import NearestNeighbors

def train_knn(X, n_neighbors=6, metric="cosine"):
    """
    Train a NearestNeighbors model.
    """
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    knn.fit(X)
    return knn