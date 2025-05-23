# Multilayer Perceptron Ensembles in a Truly Sparse Context 

This repository contains a basic implementation of the EDST_RG and Disjoint TSE. All code was implemented in Python 3.9.15. 

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all required packages in the requirements.txt file. It is recommended to create a new virtual environment, unpack the zipped code folder there, and then install all required packages to prevent any future clashes with system-wide dependencies. For users with python2 installed in the operating system, please make sure to use pip3 instead of pip.

```
pip install -r requirements.txt
```

## Usage
Running `main.py` creates all required folders for training and the storage of results. There are many options to parse arguments to the script. Without parsing arguments for Regular Disjoint TSE or Disjoint TSE, the EDST_RG as presented in the paper will be executed without the Comprehensive Refinement Phase. Please refer to all parsing options in `main.py` to try out different parameter configurations. By default, 5 submodels are trained for a total of 350 epochs.

```bash
# Experiment with EDST_RG
python3 main.py --dataset cifar10 --storage_folder cifar_experiment_EDST_RG

# Experiment with Regular Disjoint TSE
python3 main.py --dataset cifar10 --storage_folder cifar_experiment_EDST_RG --reg_disjoint 1

# Experiment with Fully Disjoint TSE
python3 main.py --dataset cifar10 --storage_folder cifar_experiment_FD_TSE --fully_disjoint 1

```

In order to actually construct the ensemble after training all subnetworks, please run the `ensemble_construction.py` file. This script requires two arguments to be parsed: ```--dataset``` and ```--storage_folder```. Parsing the arguments as passed to ```main.py``` will construct the ensemble and print the performance. Constructing the ensemble for the first example above (after training) would be done by running the following command:

```bash
python3 ensemble_construction.py --dataset cifar10 --storage_folder cifar_experiment_EDST_RG
```


## License

[MIT](https://choosealicense.com/licenses/mit/)