import numpy as np
import pandas as pd


def forward_pass_relu(X, W1, b1, W2, b2):
    """
    Forward pass for a single layer BNN.
    """
    pre_act_1 = X @ W1 + b1.reshape(1, -1)
    #pre_hidden += b1.reshape(1, -1)
    post_act_1 = np.maximum(0, pre_act_1)
    ouput = post_act_1 @ W2 + b2.reshape(1, -1)
    return ouput

def forward_pass_tanh(X, W1, b1, W2, b2):
    """
    Forward pass for a single layer BNN.
    """
    pre_act_1 = X @ W1 + b1.reshape(1, -1)
    #pre_hidden += b1.reshape(1, -1)
    post_act_1 = np.tanh(pre_act_1)
    ouput = post_act_1 @ W2 + b2.reshape(1, -1)
    return ouput

def local_prune_weights(weights, sparsity_level, index_to_prune=0):
    """
    Apply pruning to only one weight matrix in a list, specified by index.

    Parameters:
    - weights: list of np.ndarray (e.g., [W1, W2])
    - sparsity_level: fraction of weights to prune (0.0 to 1.0)
    - index_to_prune: which weight matrix to prune in the list

    Returns:
    - list of masks (one for each weight matrix)
    """
    masks = [np.ones_like(W) for W in weights]

    W = weights[index_to_prune]
    flat = np.abs(W.flatten())
    num_to_prune = int(np.floor(sparsity_level * flat.size))

    if num_to_prune > 0:
        idx = np.argpartition(flat, num_to_prune)[:num_to_prune]
        mask_flat = np.ones_like(flat, dtype=bool)
        mask_flat[idx] = False
        masks[index_to_prune] = mask_flat.reshape(W.shape).astype(float)

    return masks
