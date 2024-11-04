import argparse
import sys
import torch

DEVICE = 'cpu'
TORCH_TYPE = torch.float64
MC_SAMPLES = 512


def get_args():
    parser = argparse.ArgumentParser(description='Define parameter for the BO problem')

    parser.add_argument('-acq_m', '--acquisition_method', help='choose the methods to use to generate new points', default='nsma', type=str, choices=['qei', 'qlogei', 'qucb', 'nsma', 'nsgaii', 'random'])
    parser.add_argument('--seeds', help='choose the list of seeds for the experiments', nargs='+', default=[42], type=int)
    parser.add_argument('--parallel_mode', help='enable parallel mode', action='store_true', default=False)
    parser.add_argument('--n_initial_points', help='the number of initial points to start the BO loop', default=10, type=int)
    parser.add_argument('--noise', default=0.01, type=float, help="noise level of single task GP") # for numerical stability
    parser.add_argument(
        '--selection_type', default="X", type=str, choices=["X", "F"],  # X: clustering in variable space; F: clustering in objectives space
        help="type of selection from the Pareto front"
    )
    parser.add_argument('--exp_name', type=str, default="", help="name of the experiment")
    parser.add_argument('--n_batch', default=20, type=int, help="total number of iterations")
    parser.add_argument('--batch_size', default=3, type=int, help="number of candidtate per iteration")
    parser.add_argument('--max_it', default=20, type=int, help="number of iterations for multi-objective optimization algorithms")
    parser.add_argument('--ucb_beta', type=float, default=1.732, help="ucb beta")  # following Wilson et al. 2017
    parser.add_argument('--function_name', type=str, default="ALL", choices=[
        "Rastrigin", "Hartmann", "EggHolder", "Rosenbrock",
        "Ackley", "Alpine01", "Branin", "Schwefel",
        "Levy", "Shekel", "Michalewicz", "SixHumpCamel", 
        "Bukin", "HolderTable", "StyblinskiTang", "ALL"
    ], help="function optimize, the default optimize all the functions")
    parser.add_argument('--verbose', help='enable verbose', action='store_true', default=False)

    return parser.parse_args(sys.argv[1:])


def args_preprocessing(args):
    assert args.n_initial_points > 0
    assert args.n_batch > 0
    assert args.batch_size > 0
    assert args.noise > 0
    assert args.max_it > 0

    if args.verbose:
        print("Method to search candidates is: ", args.acquisition_method)
        if args.acquisition_method in ['nsma', 'nsgaii']:
            print("Selection Type is: ", args.selection_type)

        print("Number of candidates per iteration is: ", args.batch_size)
        print("Number of Iterations: ", args.n_batch)
        print("Noise: ", args.noise)
        print("The list of seed to use is: ", args.seeds)

    return args.acquisition_method, args.seeds, args.parallel_mode, args.n_initial_points, args.n_batch, args.batch_size, args.noise, args.max_it, args.selection_type, args.ucb_beta, args.function_name, args.verbose
