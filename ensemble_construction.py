#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from MLPs.MLP_TSE_Basic import load_sparse_model, store_results
from utils.load_data import load_cifar10_data, load_gp_data, load_higgs_data
from utils.general_utils import load_results, load_model_results, sorted_nicely

import argparse
import os
import numpy as np
from pathlib import Path
from prettytable import PrettyTable


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Truly Sparse Ensemble Example')
    parser.add_argument('--dataset', type=str, default='gp',
                        help='Dataset to use for training and evaluation')
    parser.add_argument('--storage_folder', type=str, default=None)

    # Collect parsed arguments
    args = parser.parse_args()

    # Initiate results dict for storing results
    results_dict = {}

    # Load data
    if args.dataset == 'cifar10':
        train_data, _, test_data = load_cifar10_data(50000, 10000)
    elif args.dataset == 'gp':
        train_data, _, test_data = load_gp_data(7850, 1960)

    results_dir = Path().absolute() / 'results' / args.dataset / args.storage_folder
    all_folders = os.listdir(results_dir)
    all_folders = sorted_nicely(all_folders)

    for eps_dir in all_folders:
        if not eps_dir.startswith('e'):
            all_folders.remove(eps_dir)
            continue
        else:
            eps_direc = results_dir / eps_dir
            eps_folders = os.listdir(results_dir / eps_dir)
            for model_folder in eps_folders:

                if not model_folder.startswith('model'):
                    if model_folder.startswith('results'):
                        json_file = os.listdir(results_dir / eps_dir / 'results')[0]
                        json_results = load_model_results(results_dir / eps_dir / 'results' / json_file)
                    eps_folders.remove(model_folder)

        model_files = sorted_nicely(eps_folders)

        all_folds_preds = []
        labels = []
        val_acc = []
        ce_loss = []

        for file in range(0, len(model_files)):
            submodel = load_sparse_model(eps_direc, model_files[file])

            # Call the get metrics function on the initiated model and append values to placeholders
            indi_acc, indi_loss, indi_pred, target = submodel.get_metrics(test_data[0], test_data[1])
            val_acc.append(indi_acc)
            ce_loss.append(indi_loss)
            all_folds_preds.append(indi_pred)
            labels.append(target)

            # Take the mean and std of stacked accuracies and losses
            individual_acc_mean = np.array(val_acc).mean(axis=0)
            individual_acc_std = np.array(val_acc).std(axis=0)

            # Stack the predications of all submodels and average the Softmax outputs.
            output_mean = np.mean(np.stack(all_folds_preds, axis=0), axis=0)

            # Apply argmax to get index of actual prediction
            submodel_pred = np.argmax(output_mean, axis=1)

            # Apply argmax to target to get index of correct index and compare
            submodel_correct = (submodel_pred == np.argmax(target, axis=1)).sum()
            ensemble_accuracy = round(submodel_correct / float(target.shape[0]), 3)

            results_dict[f'{eps_dir}'] = {'Ensemble_accuracy': round(ensemble_accuracy, 3),
                                          'Average_accuracy': round(individual_acc_mean, 3),
                                          'Average_sd': round(individual_acc_std, 4),
                                          'Sparsity': f"{1 - float(json_results['sparsity']):3.2f}",
                                          '# Connections': json_results['num_connections'],
                                          'Training time [min]': round(json_results['time_min'], 3)
                                          }

    # Create table of all results
    table = [['Model name', 'Ensemble accuracy', 'Average accuracy', 'Average sd', 'Sparsity',
              '# Connections', 'Training time [min]']]
    for k in results_dict.keys():
        row = list(results_dict[k].values())
        row.insert(0, k)
        table.append(row)

    tab = PrettyTable(table[0])
    tab.add_rows(table[1:])

    print(f'\nDataset: {args.dataset}')
    print("\n", tab)


if __name__ == '__main__':
    main()
