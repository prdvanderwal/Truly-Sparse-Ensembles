# imports
from scipy.sparse import coo_matrix, load_npz, save_npz
from utils.sparse_utils import *
from utils.general_utils import *
from utils.nn_functions import *

import datetime
import os
import sys

from pathlib import Path

import numpy as np

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stderr = stderr


class SET_MLP:
    def __init__(self, dimensions, activations, epsilon=20, weight_init='he_uniform', loading=False, path=None,
                 reg_disjoint=False, fully_disjoint=False):
        """
        :param dimensions: (tpl/ list) Dimensions of the neural net. (input, hidden layer, output)
        :param activations: (tpl/ list) Activations functions.
        Example of three hidden layer with
        - 3312 input features
        - 3000 hidden neurons
        - 3000 hidden neurons
        - 3000 hidden neurons
        - 5 output classes
        layers -->    [1,        2,     3,     4,     5]
        ----------------------------------------
        dimensions =  (3312,     3000,  3000,  3000,  5)
        activations = (          Relu,  Relu,  Relu,  Sigmoid)
        :param epsilon: (int) hyperparamter for the Erdos-Renyi initialization of sparse weight matrices.
        :param weight_init: (str) initialization method (he_uniform, xavier, or normal).
        :param loading: (bool) load existing model when True (otherwise initialize new sparse model).
        :param path: (path object) path of current working directory
        :param sd: (float) standard deviation for Distance TSE
        """

        self.n_layers = len(dimensions)
        self.dropout_rate = 0.  # dropout rate
        self.learning_rate = None
        self.momentum = None
        self.weight_decay = None
        self.epsilon = epsilon  # control the sparsity level as discussed in the paper
        self.zeta = None  # the fraction of the weights removed
        self.dimensions = dimensions
        self.weight_init = weight_init
        self.importance_pruning = False
        self.path = path
        self.sub_model_index = 0
        self.loss = CrossEntropy()

        self.training_time = 0
        self.testing_time = 0
        self.evolution_time = 0

        self.blocked = set()
        self.current_epoch = 0

        self.last_weights = None
        self.all_weights = get_all_weights(dimensions)

        self.reg_disjoint = reg_disjoint
        self.fully_disjoint = fully_disjoint

        if self.reg_disjoint or self.fully_disjoint:
            self.blocked_weights = {}
            for set_index in range(1, self.n_layers - 1):
                self.blocked_weights[set_index] = set()

        # Weights and biases are initiated by index. For a one hidden layer net you will have a w[1] and w[2]
        self.w = {}
        self.b = {}
        self.pdw = {}
        self.pdd = {}

        if loading:
            weights_dir = f'{self.path}/weights'
            biases_dir = f'{self.path}/biases'
            self.activations = {}
            for i in range(len(dimensions) - 1):
                self.w[i + 1] = load_npz(f'{weights_dir}/weights_{i + 1}.npz')
                self.b[i + 1] = np.load(f'{biases_dir}/biases_{i + 1}.npy')
                self.activations[i + 2] = activations[i]
        else:
            # Activations are also initiated by index. For the example we will have activations[2] and activations[3]
            self.activations = {}
            for i in range(len(dimensions) - 1):
                if self.weight_init == 'normal':
                    self.w[i + 1] = createSparseWeights_II(self.epsilon, dimensions[i],
                                                           dimensions[i + 1])  # create sparse weight matrices
                else:
                    self.w[i + 1] = create_sparse_weights(self.epsilon, dimensions[i], dimensions[i + 1],
                                                          weight_init=self.weight_init)  # create sparse weight matrices
                self.b[i + 1] = np.zeros(dimensions[i + 1], dtype='float32')
                self.activations[i + 2] = activations[i]

    def _feed_forward(self, x, drop=False):
        """
        Execute a forward feed through the network.
        :param x: (array) Batch of input data vectors.
        :return: (tpl) Node outputs and activations per layer. The numbering of the output is equivalent to the layer numbers.
        """
        # w(x) + b
        z = {}

        # activations: f(z)
        a = {1: x}  # First layer has no activations as input. The input x is the input.
        masks = {}

        for i in range(1, self.n_layers):
            z[i + 1] = a[i] @ self.w[i] + self.b[i]
            a[i + 1] = self.activations[i + 1].activation(z[i + 1])
            if drop:
                if i < self.n_layers - 1:
                    # apply dropout
                    a[i + 1], keep_mask = dropout(a[i + 1], self.dropout_rate)
                    masks[i + 1] = keep_mask

        return z, a, masks

    def _back_prop(self, z, a, masks, y_true):
        """
        The input dicts keys represent the layers of the net.
        a = { 1: x,
              2: f(w1(x) + b1)
              3: f(w2(a2) + b2)
              4: f(w3(a3) + b3)
              5: f(w4(a4) + b4)
              }
        :param z: (dict) w(x) + b
        :param a: (dict) f(z)
        :param y_true: (array) One hot encoded truth vector.
        :return:
        """
        keep_prob = 1.
        if self.dropout_rate > 0:
            keep_prob = np.float32(1. - self.dropout_rate)

        # Determine partial derivative and delta for the output layer.
        # delta output layer
        delta = self.loss.delta(y_true, a[self.n_layers])

        dw = coo_matrix(self.w[self.n_layers - 1], dtype='float32')

        # compute backpropagation updates
        backpropagation_updates_numpy(a[self.n_layers - 1], delta, dw.row, dw.col, dw.data)

        update_params = {
            self.n_layers - 1: (dw.tocsr(), np.mean(delta, axis=0))
        }

        # In case of three layer net will iterate over i = 2 and i = 1
        # Determine partial derivative and delta for the rest of the layers.
        # Each iteration requires the delta from the previous layer, propagating backwards.
        for i in reversed(range(2, self.n_layers)):
            # dropout for the backpropagation step
            if keep_prob != 1:
                delta = (delta @ self.w[i].transpose()) * self.activations[i].prime(z[i])
                delta = delta * masks[i]
                delta /= keep_prob
            else:
                delta = (delta @ self.w[i].transpose()) * self.activations[i].prime(z[i])

            dw = coo_matrix(self.w[i - 1], dtype='float32')

            # compute backpropagation updates
            backpropagation_updates_numpy(a[i - 1], delta, dw.row, dw.col, dw.data)

            update_params[i - 1] = (dw.tocsr(), np.mean(delta, axis=0))
        for k, v in update_params.items():
            self._update_w_b(k, v[0], v[1])

    def _update_w_b(self, index, dw, delta):
        """
        Update weights and biases.
        :param index: (int) Number of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error.
        """

        # perform the update with momentum
        if index not in self.pdw:
            self.pdw[index] = - self.learning_rate * dw
            self.pdd[index] = - self.learning_rate * delta
        else:
            self.pdw[index] = self.momentum * self.pdw[index] - self.learning_rate * dw
            self.pdd[index] = self.momentum * self.pdd[index] - self.learning_rate * delta

        self.w[index] += self.pdw[index] - self.weight_decay * self.w[index]
        self.b[index] += self.pdd[index] - self.weight_decay * self.b[index]

    def fit(self, x, y_true, loss, batch_size, store_dir, learning_rate=1e-3, momentum=0.9,
            weight_decay=0.0002, zeta=0.3, dropoutrate=0., testing=True, last_epoch=False,
            evolution_frequency=1, first_model_epoch=1000):
        """
        :param x: (array) Containing parameters
        :param y_true: (array) Containing one hot encoded labels.
        :return (array) A 2D array of metrics (epochs, 3).
        """
        if not x.shape[0] == y_true.shape[0]:
            raise ValueError("Length of x and y arrays don't match")

        # Initiate the loss object with the final activation function
        self.loss = loss()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.zeta = zeta
        self.dropout_rate = dropoutrate

        maximum_accuracy = 0

        # Shuffle the data
        seed = np.arange(x.shape[0])
        np.random.shuffle(seed)
        x_ = x[seed]
        y_ = y_true[seed]
        self.current_epoch += 1

        # training
        t1 = datetime.datetime.now()

        for j in range(x.shape[0] // batch_size):
            k = j * batch_size
            l = (j + 1) * batch_size
            z, a, masks = self._feed_forward(x_[k:l], True)

            self._back_prop(z, a, masks, y_[k:l])

        t2 = datetime.datetime.now()

        print("Training time: ", t2 - t1)
        self.training_time += (t2 - t1).seconds

        # test model performance on the test data at each epoch
        # this part is useful to understand model performance and can be commented for production settings

        t5 = datetime.datetime.now()

        if not last_epoch and self.current_epoch % evolution_frequency == 0:  # do not change connectivity pattern after the last epoch
            if self.fully_disjoint and self.current_epoch > first_model_epoch:
                self.weights_evolution_IIII()
            else:
                self.weights_evolution_II()  # this implementation has the same behaviour as the one above, but it is much faster.
        t6 = datetime.datetime.now()
        print("Weights evolution time ", t6 - t5)

    def evaluate(self, x_test, y_test):
        accuracy_test, _ = self.predict(x_test, y_test)
        return accuracy_test

    def get_metrics(self, x_test, y_test):
        accuracy_test, activations_test, target = self.predict_v2(x_test, y_test)
        loss_test = self.loss.loss(y_test, activations_test)
        return accuracy_test, loss_test, activations_test, target

    def save_sparse_model(self):

        # Create folder to store model within experiment folder

        model_path = create_directories(self.path / f'model_{self.sub_model_index}')

        weight_path = create_directories(model_path / 'weights')
        biases_path = create_directories(model_path / 'biases')
        activations_path = create_directories(model_path / 'activations')

        for key in self.activations.keys():
            pickle_store(f'{activations_path}/activations_{key}', self.activations[key])

        # Store weights
        for key in self.w.keys():
            save_npz(f'{weight_path}/weights_{key}', self.w[key])

        # Store biases
        for key in self.b.keys():
            np.save(f'{biases_path}/biases_{key}', self.b[key])

        # Store other values
        init_dict = {'epsilon': self.epsilon, 'len_dim': self.dimensions}
        store_results(f'{model_path}/init.json', init_dict)

    def weights_evolution_II(self, epoch=0, escape_rate=False):
        # this represents the core of the SET procedure. It removes the weights closest to zero in each layer and add new random weights
        if escape_rate:
            temp_zeta = self.zeta
            self.zeta = escape_rate

        for i in range(1, self.n_layers - 1):

            # converting to COO form - Added by Amar
            wcoo = self.w[i].tocoo()
            vals_w = wcoo.data
            rows_w = wcoo.row
            cols_w = wcoo.col

            pdcoo = self.pdw[i].tocoo()
            vals_pd = pdcoo.data
            rows_pd = pdcoo.row
            cols_pd = pdcoo.col

            values = np.sort(self.w[i].data)
            first_zero_pos = find_first_pos(values, 0)
            last_zero_pos = find_last_pos(values, 0)

            largest_negative = values[int((1 - self.zeta) * first_zero_pos)]
            smallest_positive = values[
                int(min(values.shape[0] - 1, last_zero_pos + self.zeta * (values.shape[0] - last_zero_pos)))]

            # remove the weights (W) closest to zero and modify PD as well
            vals_w_new = vals_w[(vals_w > smallest_positive) | (vals_w < largest_negative)]
            rows_w_new = rows_w[(vals_w > smallest_positive) | (vals_w < largest_negative)]
            cols_w_new = cols_w[(vals_w > smallest_positive) | (vals_w < largest_negative)]

            new_w_row_col_index = np.stack((rows_w_new, cols_w_new), axis=-1)
            old_pd_row_col_index = np.stack((rows_pd, cols_pd), axis=-1)

            new_pd_row_col_index_flag = array_intersect(old_pd_row_col_index,
                                                        new_w_row_col_index)  # careful about order

            vals_pd_new = vals_pd[new_pd_row_col_index_flag]
            rows_pd_new = rows_pd[new_pd_row_col_index_flag]
            cols_pd_new = cols_pd[new_pd_row_col_index_flag]

            self.pdw[i] = coo_matrix((vals_pd_new, (rows_pd_new, cols_pd_new)),
                                     (self.dimensions[i - 1], self.dimensions[i])).tocsr()

            # add new random connections
            keep_connections = np.size(rows_w_new)
            length_random = vals_w.shape[0] - keep_connections
            if self.weight_init == 'normal':
                random_vals = np.random.randn(length_random) / 10  # to avoid multiple whiles, can we call 3*rand?
            else:
                if self.weight_init == 'he_uniform':
                    limit = np.sqrt(6. / float(self.dimensions[i - 1]))
                if self.weight_init == 'xavier':
                    limit = np.sqrt(6. / (float(self.dimensions[i - 1]) + float(self.dimensions[i])))
                random_vals = np.random.uniform(-limit, limit, length_random)

            # adding  (wdok[ik,jk]!=0): condition
            while length_random > 0:
                ik = np.random.randint(0, self.dimensions[i - 1], size=length_random, dtype='int32')
                jk = np.random.randint(0, self.dimensions[i], size=length_random, dtype='int32')

                random_w_row_col_index = np.stack((ik, jk), axis=-1)
                random_w_row_col_index = np.unique(random_w_row_col_index,
                                                   axis=0)  # removing duplicates in new rows&cols
                oldW_row_col_index = np.stack((rows_w_new, cols_w_new), axis=-1)

                unique_flag = ~array_intersect(random_w_row_col_index,
                                               oldW_row_col_index)  # careful about order & tilda

                ik_new = random_w_row_col_index[unique_flag][:, 0]
                jk_new = random_w_row_col_index[unique_flag][:, 1]
                # be careful - row size and col size needs to be verified
                rows_w_new = np.append(rows_w_new, ik_new)
                cols_w_new = np.append(cols_w_new, jk_new)

                length_random = vals_w.shape[0] - np.size(rows_w_new)  # this will constantly reduce lengthRandom

            # adding all the values along with corresponding row and column indices - Added by Amar
            vals_w_new = np.append(vals_w_new, random_vals)  # be careful - we can add to an existing link ?
            # vals_pd_new = np.append(vals_pd_new, zero_vals) # be careful - adding explicit zeros - any reason??
            if vals_w_new.shape[0] != rows_w_new.shape[0]:
                print("not good")
            self.w[i] = coo_matrix((vals_w_new, (rows_w_new, cols_w_new)),
                                   (self.dimensions[i - 1], self.dimensions[i])).tocsr()

        # Reset value of zeta for regular SET procedure
        if escape_rate:
            self.zeta = temp_zeta

    def weights_evolution_IIII(self, epoch=0, regular_update=True, escape_rate=False):
        if escape_rate:
            temp_zeta = self.zeta
            self.zeta = escape_rate

        for i in range(2, self.n_layers - 1): # only from layer 2 onwards bc high sparsity at the start
            # uncomment line below to stop evolution of dense weights more than 80% non-zeros
            # if self.w[i].count_nonzero() / (self.w[i].get_shape()[0]*self.w[i].get_shape()[1]) < 0.8:

            # converting to COO form
            wcoo = self.w[i].tocoo()
            vals_w = wcoo.data
            rows_w = wcoo.row
            cols_w = wcoo.col

            # Add tuples of the submodel to the blocked set. Adding element has complexity O(n).
            if not regular_update:
                for indx in range(0,len(rows_w)):
                    self.blocked_weights[i].add((rows_w[indx], cols_w[indx]))
                print(f'Total set size: {len(self.blocked_weights[i])}')

            pdcoo = self.pdw[i].tocoo()
            vals_pd = pdcoo.data
            rows_pd = pdcoo.row
            cols_pd = pdcoo.col
            # print("Number of non zeros in W and PD matrix before evolution in layer",i,[np.size(valsW), np.size(valsPD)])
            values = np.sort(self.w[i].data)
            first_zero_pos = find_first_pos(values, 0)
            last_zero_pos = find_last_pos(values, 0)

            largest_negative = values[int((1 - self.zeta) * first_zero_pos)]
            smallest_positive = values[
                int(min(values.shape[0] - 1, last_zero_pos + self.zeta * (values.shape[0] - last_zero_pos)))]

            # remove the weights (W) closest to zero and modify PD as well
            vals_w_new = vals_w[(vals_w > smallest_positive) | (vals_w < largest_negative)]
            rows_w_new = rows_w[(vals_w > smallest_positive) | (vals_w < largest_negative)]
            cols_w_new = cols_w[(vals_w > smallest_positive) | (vals_w < largest_negative)]

            new_w_row_col_index = np.stack((rows_w_new, cols_w_new), axis=-1)
            old_pd_row_col_index = np.stack((rows_pd, cols_pd), axis=-1)

            new_pd_row_col_index_flag = array_intersect(old_pd_row_col_index,
                                                        new_w_row_col_index)  # careful about order

            vals_pd_new = vals_pd[new_pd_row_col_index_flag]
            rows_pd_new = rows_pd[new_pd_row_col_index_flag]
            cols_pd_new = cols_pd[new_pd_row_col_index_flag]

            self.pdw[i] = coo_matrix((vals_pd_new, (rows_pd_new, cols_pd_new)),
                                     (self.dimensions[i - 1], self.dimensions[i])).tocsr()

            keep_connections = np.size(rows_w_new)
            length_random = vals_w.shape[0] - keep_connections

            # add new random connections --> first determine indices to know how many new values can be grown
            if self.weight_init == 'normal':
                random_vals = np.random.randn(length_random) / 10  # to avoid multiple whiles, can we call 3*rand?
            else:
                if self.weight_init == 'he_uniform':
                    limit = np.sqrt(6. / float(self.dimensions[i - 1]))
                if self.weight_init == 'xavier':
                    limit = np.sqrt(6. / (float(self.dimensions[i - 1]) + float(self.dimensions[i])))
                random_vals = np.random.uniform(-limit, limit, length_random)

            # adding  (wdok[ik,jk]!=0): condition
            while length_random > 0:
                # print(f"dim i-1: {self.dimensions[i-1]}\n dim i: {self.dimensions[i]} \n Length random: {length_random}")
                ik = np.random.randint(0, self.dimensions[i - 1], size=length_random, dtype='int32')
                jk = np.random.randint(0, self.dimensions[i], size=length_random, dtype='int32')

                random_w_row_col_index = np.stack((ik, jk), axis=-1)
                random_w_row_col_index = np.unique(random_w_row_col_index,
                                                   axis=0)  # removing duplicates in new rows&cols

                layer_weights = self.w[i].getnnz()
                total_weights = self.dimensions[i-1] * self.dimensions[i]
                density_level = layer_weights / total_weights

                if density_level < 0.25:
                    to_remove = []
                    for rci in range(0, len(random_w_row_col_index)):
                        # if tuple(random_w_row_col_index[rci]) in self.blocked: # Complexity lookupt is O(1) (instant)
                        if tuple(random_w_row_col_index[rci]) in self.blocked_weights[i]:
                            to_remove.append(list(random_w_row_col_index[rci]))
                        # else:
                        #     self.blocked.add((tuple(random_w_row_col_index[rci])))
                    list_form = random_w_row_col_index.tolist()
                    for k in to_remove:
                        try:
                            list_form.remove(k)
                        except ValueError:
                            print("something went wrong")
                    random_w_row_col_index = np.array(list_form, dtype='int32')
                else:
                    print(f'The density for layer {i} was {density_level} and no attempt was made to find a disjoint matrix')

                oldW_row_col_index = np.stack((rows_w_new, cols_w_new), axis=-1)

                # If all suggested connections are already in the set, the list is empty causing array_intersect to
                # throw an error as the shape is (0,). For this reason we catch this specific error.
                try:
                    unique_flag = ~array_intersect(random_w_row_col_index,
                                                   oldW_row_col_index)  # careful about order & tilda

                    ik_new = random_w_row_col_index[unique_flag][:, 0]
                    jk_new = random_w_row_col_index[unique_flag][:, 1]
                    # be careful - row size and col size needs to be verified
                    rows_w_new = np.append(rows_w_new, ik_new)
                    cols_w_new = np.append(cols_w_new, jk_new)

                    length_random = vals_w.shape[0] - np.size(rows_w_new)  # this will constantly reduce lengthRandom
                except ValueError as e:
                    if str(e) != "not enough values to unpack (expected 2, got 1)":
                        raise
                    else:
                        continue

            # adding all the values along with corresponding row and column indices - Added by Amar
            vals_w_new = np.append(vals_w_new, random_vals)  # be careful - we can add to an existing link ?
            # vals_pd_new = np.append(vals_pd_new, zero_vals) # be careful - adding explicit zeros - any reason??
            if vals_w_new.shape[0] != rows_w_new.shape[0]:
                print("not good")
            self.w[i] = coo_matrix((vals_w_new, (rows_w_new, cols_w_new)),
                                   (self.dimensions[i - 1], self.dimensions[i])).tocsr()

        # Reset value of zeta for regular SET procedure
        if escape_rate:
            self.zeta = temp_zeta

    def predict(self, x_test, y_test, batch_size=20):
        """
        :param x_test: (array) Test input
        :param y_test: (array) Correct test output
        :param batch_size:
        :return: (flt) Classification accuracy
        :return: (array) A 2D array of shape (n_cases, n_classes).
        """
        activations = np.zeros((y_test.shape[0], y_test.shape[1]))
        for j in range(x_test.shape[0] // batch_size):
            k = j * batch_size
            l = (j + 1) * batch_size
            _, a_test, _ = self._feed_forward(x_test[k:l], drop=False)
            activations[k:l] = a_test[self.n_layers]
        accuracy = compute_accuracy(activations, y_test)
        return accuracy, activations

    def predict_v2(self, x_test, y_test, batch_size=20):
        """
        :param x_test: (array) Test input
        :param y_test: (array) Correct test output
        :param batch_size:
        :return: (flt) Classification accuracy
        :return: (array) A 2D array of shape (n_cases, n_classes).
        :return: (array) A 1D array of target prediction
        """
        activations = np.zeros((y_test.shape[0], y_test.shape[1]))
        for j in range(x_test.shape[0] // batch_size):
            k = j * batch_size
            l = (j + 1) * batch_size
            _, a_test, _ = self._feed_forward(x_test[k:l], drop=False)
            activations[k:l] = a_test[self.n_layers]
        accuracy = compute_accuracy(activations, y_test)
        target = y_test
        return accuracy, activations, target

    def next_sub_model(self):
        self.sub_model_index += 1

    def get_nnz(self):
        total_weights = 0
        for i in range(1, self.n_layers - 1):
            total_weights += self.w[i].getnnz()
        return total_weights


def load_sparse_model(path, model_name):
    # Load the activations to pass to SET_MLP object
    activations = []
    activations_dir = f'{path}/{model_name}/activations'
    new_path = path / model_name
    files = Path(activations_dir).glob('*')
    for file in files:
        loaded_file = pickle_open(file)
        activations.append(loaded_file)
    activations = tuple(activations)

    # Load epsilon
    init = load_results(f'{path}/{model_name}/init.json')

    eps = init['epsilon']
    dimensions = tuple(init['len_dim'])

    loaded_model = SET_MLP(dimensions, activations, epsilon=eps, loading=True, path=new_path)

    return loaded_model
