import networkx as nx
import numpy as np
from scipy.linalg import eigvalsh

# Create a graph (example: simple undirected graph)
G = nx.path_graph(5)  # path graph with 5 nodes

# Compute the Laplacian matrix (as a NumPy array)
L = nx.laplacian_matrix(G).toarray()

# Compute all eigenvalues (ascending order)
eigenvalues = eigvalsh(L)  # eigvalsh is for symmetric/hermitian matrices

# Print all eigenvalues
print("Eigenvalues:", eigenvalues)

# Get the second smallest eigenvalue (Fiedler value)
fiedler_value = eigenvalues[1]
print("Second smallest eigenvalue (Fiedler value):", fiedler_value)
