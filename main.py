from scipy.sparse.linalg import eigs
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch
from numpy.linalg import svd
from scipy.linalg import orth
import SCORE

if __name__ == "__main__":
    
    # Generate a synthetic graph with specified number of communities
    K = 3
    G, A = SCORE.generate_community_graph(K)

    # Apply the SCORE method to the adjacency matrix
    predicted_labels = SCORE.SCORE_method(A, K)

    # Visualize the graph with predicted community labels
    fig, ax = plt.subplots(figsize=(8, 8))  # Create a figure and an axes object
    pos = nx.spring_layout(G)  # Compute the layout

    # Draw the graph specifying the axes object explicitly
    nx.draw(G, pos, ax=ax, node_color=predicted_labels, with_labels=True, cmap=plt.cm.rainbow)
    ax.set_title("Graph with Predicted Communities using SCORE")  # Set the title on the axes object

    plt.show()  # Display the figure

    # Plot the true community labels
    true_labels = [G.nodes[i]['community'] for i in range(G.number_of_nodes())]
    fig, ax = plt.subplots(figsize=(8, 8))  # Create a figure and an axes object
    pos = nx.spring_layout(G)  # Compute the layout

    # Draw the graph specifying the axes object explicitly
    nx.draw(G, pos, ax=ax, node_color=true_labels, with_labels=True, cmap=plt.cm.rainbow)
    ax.set_title("Graph with True Communities")  # Set the title on the axes object

    plt.show()  # Display the figure