import numpy as np
import cupy as cp
from scipy.sparse import lil_matrix


def predict(matrix_U, matrix_V, matrix_Valid, bias=None, gpu=False):
    user_item_matrix = lil_matrix(matrix_Valid)
    user_item_pairs = np.asarray(user_item_matrix.nonzero()).T

    if gpu:
        matrix_U = cp.array(matrix_U)
        matrix_V = cp.array(matrix_V)
        if bias is not None:
            bias = cp.array(bias)

    temp_U = matrix_U[user_item_pairs[:, 0], :]
    temp_V = matrix_V[user_item_pairs[:, 1], :]

    if gpu:
        prediction = cp.sum(temp_U * temp_V, axis=1)
    else:
        prediction = np.sum(temp_U * temp_V, axis=1)

    if bias is not None:
        temp_bias = bias[user_item_pairs[:, 1]]
        prediction = prediction + temp_bias

    return prediction
