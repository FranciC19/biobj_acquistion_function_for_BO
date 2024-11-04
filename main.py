import pickle
import os
import random
import numpy as np
import torch
from joblib import Parallel, delayed
from datetime import datetime

from utils.utilities import pickle_folder_initialization
from utils.args_manager import get_args, args_preprocessing
from utils.config import FUNCTIONS, DIM
from optimize import run_seed


if __name__ == '__main__':

    # Get Command Line Argumrents
    args = get_args()
    acq_m, seeds, parallel_mode, n_initial_points, n_batch, batch_size, noise, max_it, selection_type, ucb_beta, function_name, verbose = args_preprocessing(args)

    date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    folder_exp = args.exp_name + "-" + date

    # Create Result Folder if it does not exists
    if not os.path.exists("experiments"):
        os.mkdir("experiments")

    pickle_folder_initialization(folder_exp)

    args_file = open(os.path.join('experiments', folder_exp, 'params.csv'), 'w')
    for key in args.__dict__.keys():
        if type(args.__dict__[key]) == float:
            args_file.write('{};{}\n'.format(key, str(round(args.__dict__[key], 10)).replace('.', ',')))
        else:
            args_file.write('{};{}\n'.format(key, args.__dict__[key]))
    args_file.close()

    if function_name != "ALL":
        # optimize the chosen function
        function_to_evaluate = [FUNCTIONS[function_name]]
    else:
        # optimize all the functions
        function_to_evaluate = FUNCTIONS.values()

    for function in function_to_evaluate:

        if verbose:
            print()

        for dim in DIM[function.name()]:

            random.seed(16007)
            np.random.seed(16007)
            torch.manual_seed(16007)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed_all(16007)
            os.environ['PYTHONHASHSEED'] = str(16007)

            fun = function(dim)

            best_observed_history = []
            new_opt_x_history = []

            # Run Method

            if not parallel_mode:
                for k, seed in enumerate(seeds):
                    
                    if verbose:
                        print('\n Function: {}, Dim: {}, Method: {}, Seed: {}/{}, Current Seed: {}'.format(
                            function.name(), dim,
                            acq_m, k + 1,
                            len(seeds), seed
                        ))
                        print("-----------------------------------------------")

                    history, current_x = run_seed(
                        seed, fun, n_initial_points, acq_m, n_batch, batch_size, 
                        noise, max_it, selection_type, ucb_beta, verbose
                    )
                    # negative sign to perform the plots 
                    history = [-h for h in history]
                    best_observed_history.append(history)
                    new_opt_x_history.append(current_x)
                    
                    if verbose:
                        print("-----------------------------------------------")
            else:
                if verbose:
                    print('\n Function: {}, Dim: {}, Method: {}'.format(
                        function.name(), dim, acq_m
                    ))
                    print("-----------------------------------------------")

                res = Parallel(n_jobs=-1)(delayed(run_seed)(
                    seed, fun, n_initial_points, acq_m, n_batch, batch_size, 
                    noise, max_it, selection_type, ucb_beta, verbose
                ) for seed in seeds)
                for k, _ in enumerate(seeds):
                    best_observed_history.append([-h for h in res[k][0]])
                    new_opt_x_history.append(res[k][1])

                if verbose:
                    print("-----------------------------------------------")

            if acq_m in ['nsma', 'nsgaii']:
                current_exp_name = '_'.join([function.name(), str(dim), acq_m + "(" + args.selection_type + ")"])

            else:
                current_exp_name = '_'.join([function.name(), str(dim), acq_m])

            pickle_folder_initialization(folder_exp, current_exp_name)

            dict_to_export = {
                'obj_f': fun.name(), 'dim': fun.dim(), 'global_obj_val': -fun.best_obj_f,
                'acquisition_method': acq_m, 'n_batch': n_batch, 'batch_size': batch_size,
                'best_observed_history': best_observed_history, 'new_opt_x_history': new_opt_x_history
            }

            out_file_pickle = open(os.path.join('experiments', folder_exp, current_exp_name, 'results.pkl'), 'wb')
            pickle.dump(dict_to_export, out_file_pickle)
            out_file_pickle.close()
