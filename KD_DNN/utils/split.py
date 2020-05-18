import scipy.sparse as sparse
import numpy as np


def seed_randomly_split(df, ratio, split_seed, shape):
    """
    Split based on a deterministic seed randomly
    """
    # Set the random seed for splitting
    np.random.seed(split_seed)

    # Randomly shuffle the data
    rows, cols, rating = df['uid'], df['iid'], df['rating']
    num_nonzeros = len(rows)
    permute_indices = np.random.permutation(num_nonzeros)
    rows, cols, rating = rows[permute_indices], cols[permute_indices], rating[permute_indices]

    # Convert to train/valid/test matrix
    idx = [int(ratio[0] * num_nonzeros), int(ratio[0] * num_nonzeros) + int(ratio[1] * num_nonzeros)]

    train = sparse.csr_matrix((rating[:idx[0]], (rows[:idx[0]], cols[:idx[0]])),
                              shape=shape, dtype='float32')

    valid = sparse.csr_matrix((rating[idx[0]:idx[1]], (rows[idx[0]:idx[1]], cols[idx[0]:idx[1]])),
                              shape=shape, dtype='float32')

    test = sparse.csr_matrix((rating[idx[1]:], (rows[idx[1]:], cols[idx[1]:])),
                             shape=shape, dtype='float32')

    return train, valid, test
