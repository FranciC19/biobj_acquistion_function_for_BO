import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from botorch.utils.transforms import unnormalize

from nsma.algorithms.memetic.nsma import NSMA
from nsma.general_utils.pareto_utils import pareto_efficient
from nsma.algorithms.genetic.nsga_ii import NSGAII

from functions.function import Function
from wrappers.extended_problem import ExtendedProblem
from utils.args_manager import DEVICE, TORCH_TYPE


class MOOAlgorithm:

    def __init__(self, alg_name: str, max_it: int):

        if alg_name == 'nsma':

            self.__alg = NSMA(max_iter=max_it,
                              max_time=None,
                              max_f_evals=None,
                              verbose=False,
                              verbose_interspace=10,
                              plot_pareto_front=False,
                              plot_pareto_solutions=False,
                              plot_dpi=100,
                              pop_size=100,
                              crossover_probability=0.9,
                              crossover_eta=20,
                              mutation_eta=20,
                              shift=np.inf,
                              crowding_quantile=0.9,
                              n_opt=5,
                              FMOPG_max_iter=5,
                              theta_for_stationarity=-1e-10,
                              theta_tol=-1e-1,
                              theta_dec_factor=10 ** (-0.5),
                              gurobi=True,
                              gurobi_method=1,
                              gurobi_verbose=False,
                              ALS_alpha_0=1,
                              ALS_delta=0.5,
                              ALS_beta=1e-4,
                              ALS_min_alpha=1e-7)

        elif alg_name == 'nsgaii':
            self.__alg = NSGAII(max_iter=max_it,
                                max_time=None,
                                max_f_evals=None,
                                verbose=False,
                                verbose_interspace=10,
                                plot_pareto_front=False,
                                plot_pareto_solutions=False,
                                plot_dpi=100,
                                pop_size=100,
                                crossover_probability=0.9,
                                crossover_eta=20,
                                mutation_eta=20)
                             
        else:
            raise NotImplementedError('Algorithm Undefined!')

    def optimize(self, selection_type: str, train_x: np.array, train_multiobj: np.array, MOO_prob: ExtendedProblem, batch_size: int, fun: Function):

        candidates_x, candidates_f, _ = self.__alg.search(train_x, train_multiobj, MOO_prob)

        efficient_point_idx = pareto_efficient(candidates_f)
        candidates_x, candidates_f = candidates_x[efficient_point_idx, :], candidates_f[efficient_point_idx, :]

        if len(candidates_x) < batch_size:
            k = 0
            selected_candidates_x = []
            for i in range(0, batch_size):

                if k < len(candidates_x):
                    selected_candidates_x.append(candidates_x[k])
                    k = k + 1

                else:
                    k = 0
                    selected_candidates_x.append(candidates_x[k])
                    k = k + 1

        elif len(candidates_x) == batch_size:
            selected_candidates_x = candidates_x

        else:
            if selection_type == "X":
                selected_candidates_x = self.cluster_X(candidates_x, batch_size)

            elif selection_type == "F":
                selected_candidates_x = self.cluster_F(candidates_x, candidates_f, batch_size)

            elif selection_type == "R":
                selected_candidates_x = self.rand_selection(candidates_x, batch_size)

        selected_candidates_x = unnormalize(torch.tensor(selected_candidates_x), fun._bounds).numpy()

        new_opt_x = torch.tensor(selected_candidates_x, device=DEVICE, dtype=TORCH_TYPE).detach()
        new_opt_obj = fun.eval(new_opt_x)

        return new_opt_x, new_opt_obj

    @staticmethod
    def cluster_X(candidates_x: np.array, batch_size: int):

        clf = KMeans(n_clusters=batch_size, n_init=10, random_state=33)
        _ = clf.fit_predict(candidates_x)
        x_centroids = clf.cluster_centers_

        return x_centroids

    @staticmethod
    def cluster_F(candidates_x: np.array, candidates_f: np.array, batch_size: int):

        scaler = StandardScaler()
        f_scaled = scaler.fit_transform(candidates_f)

        clf = KMeans(n_clusters=batch_size, n_init=10, random_state=33)
        _ = clf.fit_predict(f_scaled)

        selected_candidates_x = []
        f_centroids = clf.cluster_centers_

        for f in f_centroids:
            dist = np.linalg.norm(f - f_scaled, axis=1)  # Euclidean distance
            min_distance_index = np.argmin(dist)  # Find index of minimum distance
            closest_vector_x = candidates_x[min_distance_index]  # Get vector having minimum distance
            selected_candidates_x.append(closest_vector_x)

        return selected_candidates_x
    
    @staticmethod
    def rand_selection(candidates_x: np.array, batch_size: int):
        return candidates_x[np.random.permutation(np.arange(len(candidates_x)))[:batch_size]]
