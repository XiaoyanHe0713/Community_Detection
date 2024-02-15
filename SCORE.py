from scipy.sparse.linalg import eigs
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch
from numpy.linalg import svd
from scipy.linalg import orth

# SCORE method implementation with parameter K for number of communities
def SCORE_method(A, K):
    if K < 2:
        raise ValueError("K must be at least 2.")
    values, vectors = eigs(A, k=K, which='LR')
    vectors = np.real(vectors)
    for i in range(K):
        vectors[:, i] = vectors[:, i] / np.linalg.norm(vectors[:, i], 2)
    R = np.zeros((A.shape[0], K - 1))
    for i in range(1, K):
        R[:, i - 1] = vectors[:, i] / vectors[:, 0]
    kmeans = KMeans(n_clusters=K, random_state=0).fit(R)
    labels = kmeans.labels_
    return labels

# Function to generate a synthetic community graph
def generate_community_graph(K, nodes_per_community=10, p_in=0.8, p_out=0.05):
    N = K * nodes_per_community
    community_labels = np.repeat(np.arange(K), nodes_per_community)
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if community_labels[i] == community_labels[j]:
                if np.random.rand() < p_in:
                    A[i, j] = A[j, i] = 1
            else:
                if np.random.rand() < p_out:
                    A[i, j] = A[j, i] = 1
    G = nx.from_numpy_array(A)
    for i, label in enumerate(community_labels):
        G.nodes[i]['community'] = label
    return G, A