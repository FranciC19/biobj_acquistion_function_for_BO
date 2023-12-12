import argparse
import sys
import torch

DEVICE = 'cpu'
TORCH_TYPE = torch.float64
MC_SAMPLES = 512


def get_args():
    parser = argparse.ArgumentParser(description='Define parameter for the BO problem')

    parser.add_argument('-acq_m', '--acquisition_methods', help='choose the methods to use to generate new points',
                        default='nsma', type=str, choices=['qei', 'qucb', 'nsma', 'nsgaii'])
    parser.add_argument('--seeds', help='choose the list of seeds for the experiments', nargs='+', default=[16007, 2, 64, 3928, 1536, 42, 300, 6, 18, 101, 9741, 2023, 9050, 13, 4333, 777, 8924, 812, 21743, 42938],
                        type=int)
    parser.add_argument('--n_initial_points', help='the number of initial point to start the BO loop', default=10,
                        type=int)
    parser.add_argument('--noise', default=0.01, type=float, help="noise level of single task GP")
    parser.add_argument('--clustering_type', default="X", type=str, choices=["X", "F"],
                        help="NSMA/NSGA clustering type to perform")
    parser.add_argument('--exp_name', type=str, default="", help="name of the experiment")
    parser.add_argument('--n_batch', default=20, type=int, help="total number of iterations")
    parser.add_argument('--batch_size', default=3, type=int, help="number of candidtate per iteration")
    parser.add_argument('--max_it_na', default=20, type=int, help="number of iterations NSMA/NSGA")
    parser.add_argument('--ucb_beta', type=float, default=1.732, help="ucb beta")  # following Wilson et al. 2017
    parser.add_argument('--function_name', type=str, default="ALL", choices=["Rastrigin", "Hartmann",
                                                                             "EggHolder", "Rosenbrock",
                                                                             "Ackley", "Alpine01", "Branin", "Schwefel",
                                                                             "Levy", "Shekel", "Michalewicz",
                                                                             "SixHumpCamel", "Bukin",
                                                                             "HolderTable", "StyblinskiTang",
                                                                             "ALL"],
                        help="function optimize, the default optimize all the functions")

    return parser.parse_args(sys.argv[1:])


def args_preprocessing(args):
    assert args.n_initial_points > 0
    assert args.n_batch > 0
    assert args.batch_size > 0
    assert args.noise > 0
    assert args.max_it_na > 0

    print("Method to search candidates is: ", args.acquisition_methods)
    if 'nsma' in args.acquisition_methods or 'nsgaii' in args.acquisition_methods:
        print("Clustering Type is: ", args.clustering_type)

    print("Number of candidates per iteration is: ", args.batch_size)
    print("Number of Iterations: ", args.n_batch)
    print("Noise: ", args.noise)
    print("The list of seed to use is: ", args.seeds)

    return args.acquisition_methods, args.seeds, args.n_initial_points, args.n_batch, args.batch_size, args.noise, args.max_it_na, args.clustering_type, args.ucb_beta, args.function_name
