import numpy as np
import cupy as cp
from scipy.sparse import lil_matrix


def nll(vector_predict, vector_true, gpu=False):
    if gpu:
        vector_true = cp.array(vector_true)
        return -1 / vector_true.shape[0] * cp.sum(cp.log(1 + cp.exp(-vector_predict * vector_true)))
    else:
        return -1/vector_true.shape[0] * np.sum(np.log(1 + np.exp(-vector_predict * vector_true)))


def auc(vector_predict, vector_true, gpu=False):
    if gpu:
        vector_true = cp.array(vector_true)
        pos_indexes = cp.where(vector_true == 1)[0]
        sort_indexes = cp.argsort(vector_predict)
        rank = cp.nonzero(cp.in1d(sort_indexes, pos_indexes))[0]
        return (
                       cp.sum(rank) - len(pos_indexes) * (len(pos_indexes) + 1) / 2
               ) / (
                len(pos_indexes) * (len(vector_predict) - len(pos_indexes))
        )
    else:
        pos_indexes = np.where(vector_true == 1)[0]
        sort_indexes = np.argsort(vector_predict)
        rank = np.nonzero(np.in1d(sort_indexes, pos_indexes))[0]
        return (
                       np.sum(rank) - len(pos_indexes) * (len(pos_indexes) + 1) / 2
               ) / (
                len(pos_indexes) * (len(vector_predict) - len(pos_indexes))
        )


def evaluate(vector_Predict, matrix_Test, metric_names, gpu=False):
    global_metrics = {
        "AUC": auc,
        "NLL": nll
    }

    output = dict()

    user_item_matrix = lil_matrix(matrix_Test)
    user_item_pairs = np.asarray(user_item_matrix.nonzero()).T
    vector_Test = np.asarray(matrix_Test[user_item_pairs[:, 0], user_item_pairs[:, 1]])[0]

    results = {name: [] for name in metric_names}
    for name in metric_names:
        if gpu:
            results[name].append(
                cp.asnumpy(
                    global_metrics[name](vector_predict=vector_Predict, vector_true=vector_Test, gpu=gpu)
                ).astype('float64').flatten()[0])
        else:
            results[name].append(global_metrics[name](vector_predict=vector_Predict,
                                                      vector_true=vector_Test))

    results_summary = dict()
    for name in metric_names:
        results_summary[name] = results[name]
    output.update(results_summary)

    return output
