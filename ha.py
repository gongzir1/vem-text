import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

def hungarian_algorithm(logits):
    """
    Apply the Hungarian algorithm to logits to get permutation matrices.
    
    Args:
        logits: Tensor of shape [batch_size, n, n]
    
    Returns:
        perm_matrices: Tensor of shape [batch_size, n, n] with one-hot permutation matrices.
    """
    batch_size, n, _ = logits.size()
    perm_matrices = []

    for i in range(batch_size):
        cost_matrix = -logits[i].detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        perm_matrix = np.zeros((n, n))
        perm_matrix[row_ind, col_ind] = 1
        perm_matrices.append(perm_matrix)

    return torch.tensor(perm_matrices, dtype=torch.float32).to(logits.device)

# Initialize logits
batch_size = 3
n = 92
logits = torch.randn(batch_size, n, n)

# Apply Hungarian algorithm to get permutation matrices
perm_matrices = hungarian_algorithm(logits)

# Verify that the resulting matrices are permutation matrices
print(perm_matrices)
