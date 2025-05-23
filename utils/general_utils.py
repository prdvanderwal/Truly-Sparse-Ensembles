# imports
import re
import json
import pickle
from pathlib import Path
import numpy as np


def create_directories(path_name):
    """Creates a path if it does not exist."""

    full_path = Path(path_name)

    if not full_path.exists():
        full_path.mkdir(parents=True)
    return full_path


def store_results(path, results):
    """Store results in json format."""

    with open(path, 'w') as fp:
        json.dump(results, fp)


def load_results(path):
    """Load results in json format."""

    with open(path, 'r') as fp:
        file = json.load(fp)
        return file


def pickle_store(path, results):
    """Load results in pickle format if data cannot be serialized."""

    with open(path, 'wb') as fp:
        pickle.dump(results, fp)
        fp.close()


def pickle_open(path):
    """Store results in pickle format if data cannot be serialized."""

    with open(path, 'rb') as fp:
        file = pickle.load(fp)
        return file


def get_all_weights(dimensions):
    """
    Calculate the total number of weights if network would have been dense.
    :param dimensions: (array) array of the dimensions of each layer of the network
    """

    summed_weights = 0
    for i in range(len(dimensions)-1):
        summed_weights += (dimensions[i] * dimensions[i+1])
    return summed_weights


def print_parameters(**kwargs):
    print(f'\n{80 * "="}\n')
    for k, v in kwargs.items():
        print(f'{k}: {v}')
    print(f'\n{80 * "="}\n')


def load_model_results(path):
    results_dict = {}
    with open(path, 'r') as fp:
        file = json.load(fp)
    results_dict['sparsity'] = np.round(file['num_connections']['end'] / file['num_connections']['total'], 4)
    results_dict['time_min'] = np.round((file['train_time_sec'] / 60), 4)
    results_dict['num_connections'] = file['num_connections']['end']

    return results_dict


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)