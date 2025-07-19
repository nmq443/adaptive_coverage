import numpy as np
import networkx as nx
from scipy.linalg import eigvalsh


def lambda2(adj_mat):
    G = nx.from_numpy_array(adj_mat)

    # Compute the Laplacian matrix (as a NumPy array)
    L = nx.laplacian_matrix(G).toarray()

    # Compute all eigenvalues (ascending order)
    eigenvalues = eigvalsh(L)  # eigvalsh is for symmetric/hermitian matrices

    # Print all eigenvalues
    # print("Eigenvalues:", eigenvalues)

    # Get the second smallest eigenvalue (Fiedler value)
    fiedler_value = eigenvalues[1]
    # print("Second smallest eigenvalue (Fiedler value):", fiedler_value)
    return fiedler_value


if __name__ == "__main__":
    adj_mat = np.identity(4)
    print(lambda2(adj_mat))
