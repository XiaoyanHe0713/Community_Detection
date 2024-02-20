import numpy as np
import torch
import pandas as pd
import networkx as nx
import hypernetx as hnx
import itertools

def read_data(data_path):
    """
    Read data from a file.
    Parameters
    ----------
    data_path : str
        Path to the file containing the data.
    Returns
    -------
    data : pd.DataFrame
        Data read from the file.
    """
    # Read the data from the file
    data = pd.read_csv(data_path)
    
    return data

def df_to_hypergraph(data, nodes, edges):
    """
    Convert a DataFrame to a hypergraph.
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data
    nodes : list
        List of column names to be used as nodes
    edges : list
        List of column names to be used as edges
    Returns
    -------
    H : hypernetx.Hypergraph
        Hypergraph created from the DataFrame
    """
    # Create a list of hyperedges from the DataFrame
    hyperedges = data[edges].values.tolist()
    
    # Create the hypergraph
    H = hnx.Hypergraph(hyperedges)
    
    return H