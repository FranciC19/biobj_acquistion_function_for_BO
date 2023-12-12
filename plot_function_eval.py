import argparse
import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def get_optimum(func_name, size):
    if func_name == "Ackley":
        optimum = 0.0
    elif func_name == "Alpine01":
        optimum = 0.0
    elif func_name == "Branin":
        optimum = 0.397887
    elif func_name == "EggHolder":
        optimum = -959.6407
    elif func_name == "Hartmann":
        if size == 6:
            optimum = -3.32237
        else:
            optimum = -3.86278
    elif func_name == "Rastrigin":
        optimum = 0.0
    elif func_name == "Rosenbrock":
        optimum = 0.0
    elif func_name == "Schwefel":
        optimum = 0.0
    elif func_name == "Levy":
        optimum = 0.0
    elif func_name == "Shekel":
        optimum = -10.5363
    elif func_name == "Michalewicz":
        if size == 2:
            optimum = -1.8013
        elif size == 5:
            optimum = -4.687658
        else:
            optimum = -9.66015
    elif func_name == "SixHumpCamel":
        optimum = -1.0316
    elif func_name == "Bukin":
        optimum = 0.0
    elif func_name == "HolderTable":
        optimum = -19.2085
    elif func_name == "StyblinskiTang":
        optimum = -39.166166 * size

    return optimum


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='script to plot function evaluations')

    parser.add_argument('--exp_paths', help='list of experiments to plot', nargs='+', required=True, type=str)
    args = parser.parse_args(sys.argv[1:])

    exp_paths = args.exp_paths

    exp_dict = {}

    if not os.path.exists("plots"):  # Plot Folder to see the results
        os.mkdir("plots")

    for exp_path in exp_paths:

        results_dict = {}
        for p in os.listdir(exp_path):
            if p != "params.csv":
                func_name = p.split("_")[0]
                func_dim = p.split("_")[1]
                method = p.split("_")[2]

                result = pickle.load(open(os.path.join(exp_path, p, "results.pkl"), "rb"))

                list_of_f = np.array(result["best_observed_history"])  # shape (N_seed, N_iter)

                optimum_f = get_optimum(func_name, int(func_dim))

                results_dict[func_name + "_" + str(func_dim)] = list_of_f

        exp_dict[method] = results_dict
    
    method = list(exp_dict.keys())[0]
    
    function_to_evaluate = exp_dict[method].keys()
    plot_per_dim = {f: None for f in function_to_evaluate}

    for f in function_to_evaluate:
        plot_per_dim[f] = plt.subplots(1, 1, figsize=(7, 7))

        for exp in exp_dict.keys():
            iters = exp_dict[exp][f].shape[1]
            mean_list_of_f = np.mean(exp_dict[exp][f], axis=0)
            confidence_interval_of_f = 1.96 * np.std(exp_dict[exp][f], axis=0) / np.sqrt(exp_dict[exp][f].shape[0])

            plot_per_dim[f][1].errorbar(x=range(iters), y=mean_list_of_f, yerr=confidence_interval_of_f, label=exp,
                                        markersize=8)

            plot_per_dim[f][1].set(xlabel="Function evaluations", ylabel="BOF")
            plt.title(f)
            plt.legend()
            plt.tight_layout()
        plot_per_dim[f][0].savefig(os.path.join("plots", "{}_plot.png".format(f)))
