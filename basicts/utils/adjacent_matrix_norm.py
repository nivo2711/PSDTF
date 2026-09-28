import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse import linalg


def calculate_symmetric_normalized_laplacian(adj: np.ndarray) -> np.matrix:

    adj = sp.coo_matrix(adj)
    degree = np.array(adj.sum(1)).flatten()
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
    matrix_degree_inv_sqrt = sp.diags(degree_inv_sqrt)

    laplacian = sp.eye(adj.shape[0]) - matrix_degree_inv_sqrt.dot(adj).dot(matrix_degree_inv_sqrt).tocoo()
    return laplacian


def calculate_scaled_laplacian(adj: np.ndarray, lambda_max: int = 2, undirected: bool = True) -> np.matrix:

    if undirected:
        adj = np.maximum(adj, adj.T)

    laplacian = calculate_symmetric_normalized_laplacian(adj)

    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(laplacian, 1, which='LM')
        lambda_max = lambda_max[0]

    laplacian = sp.csr_matrix(laplacian)
    identity = sp.identity(laplacian.shape[0], format='csr', dtype=laplacian.dtype)

    scaled_laplacian = (2 / lambda_max) * laplacian - identity
    return scaled_laplacian

def calculate_symmetric_message_passing_adj(adj: np.ndarray) -> np.matrix:

    adj = adj + np.eye(adj.shape[0], dtype=np.float32)
    adj = sp.coo_matrix(adj)

    row_sum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(row_sum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    mp_adj = d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).astype(np.float32)

    return mp_adj

def calculate_transition_matrix(adj: np.ndarray) -> np.matrix:

    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(row_sum, -1)
    d_inv[np.isinf(d_inv)] = 0.0

    d_mat = sp.diags(d_inv)
    prob_matrix = d_mat.dot(adj).astype(np.float32).todense()

    return prob_matrix
