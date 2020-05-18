import numpy as np
import cupy as cp
from scipy.sparse import lil_matrix


def predict(matrix_U, matrix_V, matrix_Valid, ubias=None, ibias=None, gpu=False):
    user_item_matrix = lil_matrix(matrix_Valid)
    user_item_pairs = np.asarray(user_item_matrix.nonzero()).T

    if gpu:
        matrix_U = cp.array(matrix_U)
        matrix_V = cp.array(matrix_V)
        if ubias is not None:
            ubias = cp.array(ubias)
            ibias = cp.array(ibias)

    temp_U = matrix_U[user_item_pairs[:, 0], :]
    temp_V = matrix_V[user_item_pairs[:, 1], :]

    if gpu:
        prediction = cp.sum(temp_U * temp_V, axis=1)
    else:
        prediction = np.sum(temp_U * temp_V, axis=1)

    if ubias is not None:
        temp_ubias = ubias[user_item_pairs[:, 0]]
        temp_ibias = ibias[user_item_pairs[:, 1]]
        prediction = prediction + temp_ibias + temp_ubias
    return prediction
