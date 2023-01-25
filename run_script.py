
import os
import time
import argparse

parser = argparse.ArgumentParser(description='Run a script via sbatch or interactive mode')
parser.add_argument('--sbatch', action='store_true', help='creates sbatch for script')
parser.set_defaults(sbatch=False)
args = parser.parse_args()

import time
import csv
from itertools import product

# Utility function

def make_sbatch_params(param_dict):

    trials = [ { p : t for p, t in zip(param_dict.keys(), trial) }
                    for trial in list(product(*param_dict.values())) ]

    def trial_to_args(trial):
        arg_list = ['--' + param + ' ' + str(val) if type(val) != type(True)
                else '--' + param if val else '' for param, val in trial.items()]
        return ' '.join(arg_list)

    sbatch_params = [trial_to_args(trial) for trial in trials]

    return sbatch_params

with open('metrics.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['settings', 'best_score', 'first_guess', 'best_epoch'])


single_params = '--n_layers 5 --lr 1e-5 --wd 1e-1 --seed 10 --split_word 0.8 --load_model pretrain-save_model-n_epochs-500-split_word-0.8-n_layers-5'

param_dict = {
    'seed' : [1],
    'lr' : [1e-3],
    'n_layers' : [5],
    'pretrain' : [True],
    'save_model' : [True],
    'n_epochs' : [500],
    'split_word' : [0.8],
    'wd' : [0, 1e-3, 1e-2, 1e-1],
}

sbatch_params = make_sbatch_params(param_dict)

if args.sbatch:

    print('Submitted', len(sbatch_params), 'jobs')

    for params in sbatch_params:
        os.system('sbatch run_script.sh \'' + params + '\'')
else:
    os.system('python3 main.py ' + single_params)
