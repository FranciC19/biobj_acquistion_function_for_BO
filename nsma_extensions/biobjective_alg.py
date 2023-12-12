import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from nsma.algorithms.memetic.nsma import NSMA
from nsma.general_utils.pareto_utils import pareto_efficient
from nsma.algorithms.genetic.nsga_ii import NSGAII
from botorch.utils.transforms import unnormalize

from utils.args_manager import DEVICE, TORCH_TYPE


class BiObjectiveAlg:

    def __init__(self, alg_name, max_it_na):

        if alg_name == 'nsma':

            self.__alg = NSMA(max_iter=max_it_na,
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
            self.__alg = NSGAII(max_iter=max_it_na,
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

    def optimize(self, clustering_type, train_x,
                 train_biobj, BO_prob, batch_size, fun):

        candidates_x, candidates_f, _ = self.__alg.search(train_x, train_biobj, BO_prob)

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
            selected_candidates_x = np.array(selected_candidates_x)
            selected_candidates_x = unnormalize(torch.tensor(selected_candidates_x), fun._bounds).numpy()

        elif len(candidates_x) == batch_size:
            selected_candidates_x = candidates_x
            selected_candidates_x = unnormalize(torch.tensor(selected_candidates_x), fun._bounds).numpy()

        else:
            if clustering_type == "X":
                selected_candidates_x = self.cluster_X(candidates_x, batch_size, fun)

            elif clustering_type == "F":
                selected_candidates_x = self.cluster_F(candidates_x, candidates_f, batch_size, fun)

        new_opt_x = torch.tensor(selected_candidates_x, device=DEVICE, dtype=TORCH_TYPE).detach()
        new_opt_obj = fun.eval(new_opt_x)

        return new_opt_x, new_opt_obj

    @staticmethod
    def cluster_X(candidates_x, batch_size, fun):

        clf = KMeans(n_clusters=batch_size, n_init=10, random_state=33)
        _ = clf.fit_predict(candidates_x)
        x_centroids = clf.cluster_centers_
        selected_candidate_x = unnormalize(torch.tensor(x_centroids), fun._bounds).numpy()

        return selected_candidate_x

    @staticmethod
    def cluster_F(candidates_x, candidates_f, batch_size, fun):

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

        selected_candidates_x = unnormalize(torch.tensor(selected_candidates_x), fun._bounds).numpy()

        return selected_candidates_x
