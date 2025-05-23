# imports
import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def load_cifar10_data(n_training_samples=50000, n_testing_samples=10000, validation_split = None):

    # read CIFAR10 data
    (x, y), (x_test, y_test) = cifar10.load_data()

    y = to_categorical(y, 10)
    y_test = to_categorical(y_test, 10)
    x_train = x.astype('float32')
    x_test = x_test.astype('float32')

    index_train = np.arange(x.shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(x_test.shape[0])
    np.random.shuffle(index_test)

    if n_training_samples > len(x_train):
        n_training_samples = len(x_train)
        n_testing_samples = len(x_test)

    x_train = x_train[index_train[0:n_training_samples], :]
    y_train = y[index_train[0:n_training_samples], :]

    x_test = x_test[index_test[0:n_testing_samples], :]
    y_test = y_test[index_test[0:n_testing_samples], :]

    # Normalize data
    x_train_mean = np.mean(x_train, axis=0)
    x_train_std = np.std(x_train, axis=0)
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    x_train = x_train.reshape(-1, 32 * 32 * 3)
    x_test = x_test.reshape(-1, 32 * 32 * 3)

    if validation_split:
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=88,
                                                           train_size=(1-validation_split), shuffle=True)

        return (x_train, y_train),  (x_val, y_val), (x_test, y_test)

    return (x_train, y_train), 0, (x_test, y_test)


def load_gp_data(n_training_samples=7898, n_testing_samples=1975, validation_split = None, new=False):
    x_train = np.load(f'data/GP/x_train.npy')
    y_train = np.load(f'data/GP/y_train.npy')
    x_test = np.load(f'data/GP/x_test.npy')
    y_test = np.load(f'data/GP/y_test.npy')


    np.random.seed(78)
    index_train = np.arange(x_train.shape[0])
    np.random.shuffle(index_train)

    np.random.seed(78)
    index_test = np.arange(x_test.shape[0])
    np.random.shuffle(index_test)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    if n_training_samples > len(x_train):
        n_training_samples = len(x_train)
        n_testing_samples = len(x_test)

    x_train = x_train[index_train[0:n_training_samples], :]
    y_train = y_train[index_train[0:n_training_samples], :]

    x_test = x_test[index_test[0:n_testing_samples], :]
    y_test = y_test[index_test[0:n_testing_samples], :]

    if validation_split:
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, random_state=88,
                                                          train_size=(1 - validation_split), shuffle=True)

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    return (x_train, y_train), 0, (x_test, y_test)



