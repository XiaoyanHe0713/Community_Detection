import numpy as np
import torch
import pandas as pd
import networkx as nx
import hypernetx as hnx
import itertools
from TensorMethods import *
from sklearn.cluster import KMeans

def generate_random_uniform_hypergraph(n_nodes, n_hyperedges, nodes_per_hyperedge):
    """
    Generate a random m-uniform hypergraph with n_nodes nodes, n_hyperedges hyperedges, and max_nodes_per_hyperedge nodes per hyperedge.
    Parameters
    ----------
    n_nodes : int
        Number of nodes in the hypergraph
    n_hyperedges : int
        Number of hyperedges in the hypergraph
    max_nodes_per_hyperedge : int
        Maximum number of nodes per hyperedge
    Returns
    -------
    H : hypernetx.Hypergraph
        Random uniform hypergraph
    """
    nodes = list(range(n_nodes))
    edges = list(range(n_hyperedges))
    m = nodes_per_hyperedge

    # Generate random hyperedges
    hyperedges = {e: np.random.choice(nodes, m, replace=False) for e in edges}

    # Create the hypergraph
    H = hnx.Hypergraph(hyperedges)

    return H

def adjacency_tensor(H, n_nodes):
    """
    Compute the adjacency tensor of a m-uniform hypergraph.
    Parameters
    ----------
    H : hypernetx.Hypergraph
        Hypergraph
    Returns
    -------
    A : torch tensor
        Adjacency tensor of the hypergraph
    """
    m = len(H.edges[0])

    hyperedges = H.incidence_dict

    # Dimension of the adjacency tensor is \underbrace{n_nodes \times \cdots \times n_nodes}_{m}
    dim = [n_nodes] * m

    # Initialize the adjacency tensor
    A = torch.zeros(dim)

    # For all 1 ≤ i1, . . . , im ≤ n, set A(i1, . . . , im) = 1 if {i1, . . . , im} ∈ E, and 0 otherwise
    for i in H.edges:
        # Check if the hyperedge has the right number of nodes (m)
        hyperedge = hyperedges[i]
        if len(hyperedge) != m:
            raise ValueError("All hyperedges must have exactly {} nodes".format(m))
        
        # Set the adjacency tensor to 1 for all permutations of the hyperedge
        for perm in itertools.permutations(hyperedge):
            A[perm] = 1

    return A


def adjacency_to_hypergraph(A):
    """
    Convert an adjacency tensor to a hypergraph.
    Parameters
    ----------
    A : torch tensor
        Adjacency tensor
    Returns
    -------
    H : hypernetx.Hypergraph
        Hypergraph
    """
    # Get the number of nodes
    n_nodes = A.shape[0]

    # Get the number of dimensions
    m = len(A.shape)

    # Initialize the hyperedges
    hyperedges = dict()
    list_e = []

    # Get all the indices where A is 1
    indices = torch.nonzero(A).tolist()

    # if an index is a permutation of another index, we only need to add one of them
    # to the list of hyperedges

    # Iterate over all indices
    i = 0
    for index in indices:
            # Check if the index is a permutation of another index
            is_true = False
            for perm in itertools.permutations(index):
                if perm not in list_e:
                    list_e.append(perm)
                    is_true = True
            if is_true:
                hyperedges[i] = index
                i += 1
    
    # Create the hypergraph
    H = hnx.Hypergraph(hyperedges)

    return H

def NonUniform_to_Uniform(H):
    """
    Convert a non-uniform hypergraph to a sequence of uniform hypergraphs.
    Parameters
    ----------
    H : hypernetx.Hypergraph
        Non-uniform hypergraph
    Returns
    -------
    H_seq : list of hypernetx.Hypergraph
        Sequence of uniform hypergraphs
    """
    # Get the number of nodes
    n_nodes = len(H.nodes)

    # Classify the hyperedges into different groups based on the number of nodes in each hyperedge
    groups = dict()
    for e in H.edges:
        m = len(H.incidence_dict[e])
        if m not in groups:
            groups[m] = []
        groups[m].append(e)

    # Convert each group of hyperedges to a uniform hypergraph
    H_seq = []
    for m, edges in groups.items():
        # Create a uniform hypergraph
        incidence_dict = dict()
        for i, e in enumerate(edges):
            incidence_dict[i] = e

        # Add the uniform hypergraph to the sequence
        H_seq.append(hnx.Hypergraph(incidence_dict))

    return H_seq

def if_Uniform(H):
    """
    Check if a hypergraph is uniform.
    Parameters
    ----------
    H : hypernetx.Hypergraph
        Hypergraph
    Returns
    -------
    bool
        True if the hypergraph is uniform, False otherwise
    """
    # Classify the hyperedges into different groups based on the number of nodes in each hyperedge
    groups = dict()
    for e in H.edges:
        m = len(H.incidence_dict[e])
        if m not in groups:
            groups[m] = []
        groups[m].append(e)

    # If there is only one group, the hypergraph is uniform
    return len(groups) == 1

def Tensor_SCORE(H, K, n_iter_max=100):
    """
    Apply the SCORE method to a hypergraph.
    Parameters
    ----------
    H : hypernetx.Hypergraph
        Hypergraph
    K : int
        Number of communities
    n_iter_max : int, default is 100
        Maximum number of iterations
    Returns
    -------
    labels : list
        Predicted community labels
    """
    # Check if the hypergraph is uniform
    if not if_Uniform(H):
        H_seq = NonUniform_to_Uniform(H)

        R_hat = torch.empty(0, K - 1)
        for H in H_seq:
            A = adjacency_tensor(H, len(H.nodes))
            core, factors = tucker_decomp(A, [K] * len(A.shape), n_iter_max=n_iter_max)
            factor = factors[0]
            factor = scale_invariant(factor)

            # Concatenate the factor matrices
            R_hat = torch.cat((R_hat, factor[:, 1:]), dim=0)

            kmeans = KMeans(n_clusters=K, random_state=0).fit(R_hat)
            labels = kmeans.labels_

        return labels

    # Convert the hypergraph to an adjacency tensor
    A = adjacency_tensor(H, len(H.nodes))

    # Apply the Tucker decomposition to find the factor matrices
    core, factors = tucker_decomp(A, [K] * len(A.shape), n_iter_max=n_iter_max)

    # The adjacency tensor must be symmetric, then the factor matrices are same, so we can use the first one
    factor = factors[0]

    # Apply scale-invariant to the factor matrix
    factor = scale_invariant(factor)
    R_hat = factor[:, 1:]

    # Apply K-mean Clustering to the rows of factor matrix
    kmeans = KMeans(n_clusters=K, random_state=0).fit(R_hat)
    labels = kmeans.labels_

    return labels

def visualize_hypergraph(H, labels):
    """
    Visualize a hypergraph with predicted community labels.
    Parameters
    ----------
    H : hypernetx.Hypergraph
        Hypergraph
    labels : list
        Predicted community labels
    """
    # Create a dictionary of nodes and their community labels
    node_labels = {n: l for n, l in zip(H.nodes, labels)}

    # Draw the hypergraph with predicted community labels
    hnx.draw(H, node_labels=node_labels)