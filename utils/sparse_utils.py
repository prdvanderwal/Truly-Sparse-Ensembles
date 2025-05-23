# imports
from MLPs.MLP_TSE_Basic import *

from scipy.sparse import lil_matrix
from sklearn.metrics.pairwise import euclidean_distances
from numba import njit, prange
import numpy as np
from pathlib import Path


@njit(parallel=True, fastmath=True, cache=True)
def backpropagation_updates_numpy(a, delta, rows, cols, out):
    for i in prange(out.shape[0]):
        s = 0
        for j in range(a.shape[0]):
            s += a[j, rows[i]] * delta[j, cols[i]]
        out[i] = s / a.shape[0]


@njit(fastmath=True, cache=True)
def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


@njit(fastmath=True, cache=True)
def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx


@njit(fastmath=True, cache=True)
def compute_accuracy(activations, y_test):
    correct_classification = 0
    for j in range(y_test.shape[0]):
        if np.argmax(activations[j]) == np.argmax(y_test[j]):
            correct_classification += 1
    return correct_classification / y_test.shape[0]


@njit(fastmath=True, cache=True)
def dropout(x, rate):
    noise_shape = x.shape
    noise = np.random.uniform(0., 1., noise_shape)
    keep_prob = 1. - rate
    scale = np.float32(1 / keep_prob)
    keep_mask = noise >= rate
    return x * scale * keep_mask, keep_mask


def get_index(no):
    return np.random.randint(0, no)

def scipy_dist(w1, w2):
    distances = euclidean_distances(w1, w2)
    # print(f"Distance between matrices: {np.sum(distances)}")
    return np.sum(distances)


def createSparseWeights_II(epsilon, noRows, noCols):
    # generate an Erdos Renyi sparse weights mask
    weights = lil_matrix((noRows, noCols))
    for i in range(epsilon * (noRows + noCols)):
        weights[np.random.randint(0, noRows), np.random.randint(0, noCols)] = np.float32(np.random.randn() / 10)
    print("Create sparse matrix with ", weights.getnnz(), " connections and ",
          (weights.getnnz() / (noRows * noCols)) * 100, "% density level")
    weights = weights.tocsr()
    return weights


def create_sparse_weights(epsilon, n_rows, n_cols, weight_init):
    # He uniform initialization
    if weight_init == 'he_uniform':
        limit = np.sqrt(6. / float(n_rows))

    # Xavier initialization
    if weight_init == 'xavier':
        limit = np.sqrt(6. / (float(n_rows) + float(n_cols)))

    mask_weights = np.random.rand(n_rows, n_cols)
    prob = 1 - (epsilon * (n_rows + n_cols)) / (n_rows * n_cols)  # normal to have 8x connections

    # generate an Erdos Renyi sparse weights mask
    weights = lil_matrix((n_rows, n_cols))
    n_params = np.count_nonzero(mask_weights[mask_weights >= prob])
    weights[mask_weights >= prob] = np.random.uniform(-limit, limit, n_params)
    print("Create sparse matrix with ", weights.getnnz(), " connections and ",
          (weights.getnnz() / (n_rows * n_cols)) * 100, "% density level")
    weights = weights.tocsr()
    return weights


def array_intersect(a, b):
    # this are for array intersection
    n_rows, n_cols = a.shape
    dtype = {'names': ['f{}'.format(i) for i in range(n_cols)], 'formats': n_cols * [a.dtype]}
    return np.in1d(a.view(dtype), b.view(dtype))  # boolean return



