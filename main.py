# imports
from MLPs.MLP_TSE_Basic import *
from utils.load_data import *
from utils.general_utils import *
from utils.nn_functions import *


import datetime
import os
import time
import sys
import argparse

from pathlib import Path
import numpy as np

# Write to console
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stderr = stderr


def milestone_calculation(args):
    """
    :param args: (parser)
    :return: milestone calculation for when to end the exploration phase and the second step of the refinement phase.
    """
    individual_epoch = (args.epochs - args.epochs_explore) / args.num_models
    args.individual_epoch = individual_epoch
    reset_lr_epochs1 = []
    epoch_ = args.epochs_explore
    for epoch in range(args.num_models):
        reset_lr_epochs1.append(epoch_)
        epoch_ = epoch_ + individual_epoch
    reset_lr_epochs2 = np.array(reset_lr_epochs1) + individual_epoch / 2
    reset_lr_epochs1.pop(0)
    return np.ceil(reset_lr_epochs1), np.ceil(reset_lr_epochs2)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Truly Sparse Ensemble subnetwork training Example')

    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=350, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--epochs_explore', type=int, default=100, metavar='N',
                        help='training time of exploration phase')
    parser.add_argument('--iter', type=int, default=1,
                        help='How many times the model should be run after each other. Default=1')
    parser.add_argument('--num_models', type=int, default=5,
                        help='How many subnetworks to produce, default=3')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=32, metavar='S', help='random seed (default: 88)')
    parser.add_argument('--valid_split', type=float, default=0.2,
                        help='percentage of training data used for validation')
    parser.add_argument('--num_hidden_layers', type=int, default=3,
                        help='number of hidden layers')
    parser.add_argument('--num_neurons', type=int, default=1000,
                        help='number of neurons per hidden layers')
    parser.add_argument('--reg_disjoint', type=int, default=0, help='Set to 1 to run experiments with Regular Disjoint '
                                                                    'TSE. Note,it is not possible to run Regular '
                                                                    'Disjoint TSE and Fully Disjoint TSE '
                                                                    'simultaneously.')
    parser.add_argument('--fully_disjoint', type=int, default=0)
    parser.add_argument('--epsilon', type=int, default=20)
    parser.add_argument('--large-death-rate', type=float, default=0.80, help=' large exploration rate q.')
    parser.add_argument('--zeta', type=float, default=0.20, help='Pruning rate during SET procedure')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate during training')
    parser.add_argument('--evolution_frequency', type=int, default=2, help='Inverse of the frequency (epochs) with'
                                                                           'which weight evolution is applied during training.'
                                                                           'Increasing this hyperparamter results in '
                                                                           'less frequent topology updates')
    parser.add_argument('--dataset', type=str, default='gp',
                        help='Available dataset options: cifar10 and gp.')
    parser.add_argument('--storage_folder', type=str, default=None)
    parser.add_argument('--alpha', type=float, default=0.6, help='hyperparameter for Alternated Left ReLU. Values used'
                                                                 ' in the experiments can be found in the supplementary'
                                                                 'material.')
    parser.add_argument('--init', type=str, default='he_uniform', help='Initialization method. Options include:'
                                                                       'he_uniform, xavier, normal')

    # Collect parsed arguments
    args = parser.parse_args()

    # Calculate the milestones
    milestone1, milestone2 = milestone_calculation(args)

    # Calculate when to start with disjoint
    if args.fully_disjoint:
        first_model_epoch = milestone1[0] + 1
    else:
        first_model_epoch = float('inf')

    # Load data
    if args.dataset == 'cifar10':
        train_data, val_data, test_data = load_cifar10_data(validation_split=args.valid_split)

    elif args.dataset == 'gp':
        train_data, val_data, test_data = load_gp_data(validation_split=args.valid_split)
    else:
        print('The selected dataset is invalid. Please parse a valid dataset: cifar10 or gp')

    # Print overview of paramters
    print_parameters(Epochs=args.epochs, Exploration_Epochs=args.epochs_explore, Dataset=args.dataset,
                     num_models=args.num_models, Epsilon=args.epsilon, Zeta=args.zeta,
                     Evolution_Freq=args.evolution_frequency, Regular_disjoint=args.reg_disjoint,
                     Fully_disjoint=args.fully_disjoint,
                     Milestone_1=milestone1, Milestone_2=milestone2)

    for iteration in range(args.iter):
        print(f"\nIteration start: {iteration + 1}/{args.iter}\n")

        # Create directory for current iteration
        experiment_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        if args.storage_folder:
            work_dir = create_directories(Path().absolute() / 'results' / args.dataset / args.storage_folder / f'e{str(args.epsilon)}')
        else:
            work_dir = create_directories(Path().absolute() / 'results' / args.dataset / experiment_time)


        # Inititate truly sparse network
        model = SET_MLP((train_data[0].shape[1], args.num_neurons, args.num_neurons,
                         args.num_neurons, train_data[1].shape[1]),
                        (AlternatedLeftReLU(-1*args.alpha), AlternatedLeftReLU(args.alpha),
                         AlternatedLeftReLU(-1* args.alpha), Softmax), weight_init=args.init,
                        epsilon=args.epsilon, path=work_dir, reg_disjoint=args.reg_disjoint,
                        fully_disjoint=args.fully_disjoint)


        # Placeholder for variables
        metrics = {'best_submodel_acc': [], 'epoch_time': [], 'num_connections': {}}
        metrics['num_connections']['end'] = 0

        last_epoch = False
        best_acc = 0.0
        best_submodel_acc = 0.0

        # Start timer
        starting_time = time.time()

        # Start epoch iteration
        for epoch in range(1, args.epochs + 1):

            print(f'\n{60 * "="}\nEpoch {epoch}\n{60 * "="}')

            # If it is the last epoch, don't apply weight evolution (SET)
            if epoch == args.epochs:
                last_epoch = True

            # Filename for storage
            save_filename = f"e{args.epsilon}" \
                            f"z:{args.zeta}_sub:{args.num_models}_" \
                            f"seed:{args.seed}"

            # Create dictionary for storing metrics
            metrics[f'Epoch {epoch}'] = {}

            # Use validation data (if possible) to evaluate submodel
            if args.valid_split > 0.0:
                x_val, y_val = val_data[0], val_data[1]
            else:
                x_val, y_val = test_data[0], test_data[1]

            # Fit the model for 1 epoch
            model.fit(train_data[0], train_data[1], loss=CrossEntropy,
                      batch_size=args.batch_size,
                      store_dir=work_dir, learning_rate=args.lr,
                      momentum=args.momentum, weight_decay=0.0002, zeta=args.zeta, dropoutrate=args.dropout,
                      last_epoch=last_epoch,
                      evolution_frequency=args.evolution_frequency,
                      first_model_epoch=first_model_epoch)

            # Evaluate the model at validation or test_data
            val_acc = model.evaluate(x_val, y_val)
            if val_acc > best_acc:
                print('Saving model')
                best_acc = val_acc
                print(f'\nCurrent best accuracy: {best_acc}')
                model.save_sparse_model()

            # Logging of all results
            accuracy_test, activations_test = model.predict(test_data[0], test_data[1], batch_size=20)
            accuracy_train, activations_train = model.predict(train_data[0], train_data[1], batch_size=20)
            loss_test = model.loss.loss(test_data[1], activations_test)
            loss_train = model.loss.loss(train_data[1], activations_train)

            # Best submodel accuracy on the test set
            if accuracy_test > best_submodel_acc:
                best_submodel_acc = accuracy_test

            metrics[f'Epoch {epoch}']['Training Loss'] = np.round(loss_train, 4)
            metrics[f'Epoch {epoch}']['Testing Loss'] = np.round(loss_test, 4)
            metrics[f'Epoch {epoch}']['Training Accuracy'] = np.round(accuracy_train, 4)
            metrics[f'Epoch {epoch}']['Testing Accuracy'] = np.round(accuracy_test, 4)
            metrics[f'Epoch {epoch}']['Learning rate'] = args.lr
            metrics[f'Epoch {epoch}']['Zeta'] = args.zeta

            if epoch == args.epochs_explore:
                if args.fully_disjoint or args.reg_disjoint:
                    args.evolution_frequency *= 2
                else:
                    args.lr = 0.1
            elif epoch in milestone2:
                if args.fully_disjoint or args.reg_disjoint:
                    args.lr /= 2
                    args.evolution_frequency *= 2
                    args.zeta /= 2
                else:
                    args.lr = 0.05
            elif epoch in milestone1:
                metrics['best_submodel_acc'].append(best_submodel_acc)

                """
                Milestone one contains the epochs indices of when to start a new subnetwork. We apply the same SET
                procedure as during normal training but pass the argument of the large_death_rate.
                """

                if args.reg_disjoint or args.fully_disjoint:
                    model.weights_evolution_IIII(escape_rate=args.large_death_rate, regular_update=False)
                    args.lr *= 2
                    args.evolution_frequency /= 2
                    args.zeta *= 2
                else:
                    model.weights_evolution_II(escape_rate=args.large_death_rate)

                # Reset placeholders and learning rate
                best_acc = 0.0
                best_submodel_acc = 0.0
                args.lr = 0.1

                # Increase submodel index
                model.next_sub_model()

        end_timing = time.time()

        metrics['best_submodel_acc'].append(best_submodel_acc)
        metrics['num_connections']['end'] = model.get_nnz()
        metrics['num_connections']['total'] = model.all_weights
        metrics['epsilon'] = args.epsilon

        print(f'Total training time: {round(((end_timing - starting_time) / 60), 3)} minutes')

        metrics['train_time_sec'] = (end_timing - starting_time)

        results_path = create_directories(work_dir / 'results')
        store_results(f'{results_path}/{save_filename}.json', metrics)


if __name__ == '__main__':
    main()
