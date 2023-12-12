import os
import random
import time
import numpy as np
import torch
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP, FixedNoiseGaussianLikelihood
from botorch.utils.transforms import normalize
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
from nsma_extensions.biobjective_obj import BiObjectiveProblem
from nsma_extensions.biobjective_alg import BiObjectiveAlg
from sklearn.preprocessing import MinMaxScaler

from acq_function import AcqFunction
from utils.args_manager import MC_SAMPLES, DEVICE, TORCH_TYPE


def generate_initial_data(fun, initial_points):
    train_x = torch.add(torch.mul(torch.rand(initial_points, fun.dim(), device=DEVICE, dtype=TORCH_TYPE),
                                  torch.sub(fun.bounds[1, :], fun.bounds[0, :])), fun.bounds[0, :])
    train_obj = fun.eval(train_x)
    best_obj = train_obj.max().item()
    return train_x, train_obj, best_obj


def initialize_singletaskGP(train_x, train_obj, noise, fun, state_dict=None):
    train_x = normalize(train_x, fun._bounds)
    train_obj = torch.tensor(MinMaxScaler().fit_transform(train_obj.numpy(force=True)))

    model = SingleTaskGP(train_x,
                         train_obj,
                         likelihood=FixedNoiseGaussianLikelihood(noise=torch.ones(len(train_obj)) * noise)
                         ).to(train_x)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model


def run_seed(seed, fun, n_initial_points, acq_m, n_batch, batch_size,
             noise, max_it_na, clustering_type, ucb_beta):
    print()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]), seed=seed)

    # Generate initial data points
    train_x, train_obj, first_best_observed = generate_initial_data(fun, n_initial_points)
    best_observed_seed = [first_best_observed]
    new_opt_x_list = [train_x]

    mll, model = initialize_singletaskGP(train_x, train_obj, noise, fun)

    # Iteration for loop 
    for it_batch in range(n_batch):

        start_time = time.time()

        # Gaussian Process
        fit_gpytorch_mll(mll=mll)

        if acq_m in ['qei', 'qucb']:
            # generate random points
            train_rand_x = torch.add(torch.mul(torch.rand(100, batch_size, fun.dim(), device=DEVICE, dtype=TORCH_TYPE),
                                               torch.sub(fun.bounds[1, :], fun.bounds[0, :])), fun.bounds[0, :])
            # normalize points in bounds
            train_rand_x = normalize(train_rand_x, fun._bounds)
            # scale function 
            norm_train_obj = torch.tensor(MinMaxScaler().fit_transform(train_obj.numpy(force=True)))
            # initialize the acquisition function
            acq_f = AcqFunction(acq_m, model, fun, norm_train_obj, qmc_sampler, ucb_beta=ucb_beta)
            # optimize the acquisition function 
            new_opt_x, new_opt_obj = acq_f.optimize(train_rand_x, batch_size)


        elif acq_m in ['nsma', 'nsgaii']:
            # initialize random points 
            train_rand_x = torch.add(torch.mul(torch.rand(100, fun.dim(), device=DEVICE, dtype=TORCH_TYPE),
                                               torch.sub(fun.bounds[1, :], fun.bounds[0, :])), fun.bounds[0, :])

            # nsma/nsga optimization problem
            biobj_prob = BiObjectiveProblem(fun.dim(), model, np.zeros((fun.dim())), np.ones((fun.dim())))

            # normalize points in bounds
            train_rand_x = normalize(train_rand_x, fun._bounds).numpy(force=True)

            # scale function 
            norm_train_obj = torch.tensor(MinMaxScaler().fit_transform(train_obj.numpy(force=True)))

            # evaluation of mu and sigma on the points to be optimized by NSMA/NSGA
            train_biobj = np.array(
                [biobj_prob.evaluate_functions(train_rand_x[i, :]) for i in range(train_rand_x.shape[0])])

            # Wrapper nsma/nsga 
            biobj_alg = BiObjectiveAlg(acq_m, max_it_na)
            # NSMA must take bounds [0, 1]
            new_opt_x, new_opt_obj = biobj_alg.optimize(clustering_type, train_rand_x, train_biobj, biobj_prob,
                                                        batch_size, fun)

        else:
            raise NotImplementedError('Acquisition Method Undefined!')

        if new_opt_x is not None and new_opt_obj is not None:
            train_x = torch.cat([train_x, new_opt_x])
            train_obj = torch.cat([train_obj, new_opt_obj])

        best_observed_seed.append(train_obj.max().item())
        new_opt_x_list.append(new_opt_x)

        mll, model = initialize_singletaskGP(train_x, train_obj, noise, fun, model.state_dict())

        print("Batch {}/{} > Best F: {}; Time: {}".format(it_batch + 1, n_batch, -best_observed_seed[-1],
                                                          time.time() - start_time))

    print("\n Here is a recap:")

    # Find the the best observed value to be printed
    best_n_iter = n_batch - 1
    while best_n_iter > 0:
        if best_observed_seed[best_n_iter - 1] < best_observed_seed[best_n_iter]:
            break
        best_n_iter -= 1

    print("Best value of {} is {} and has been seen first at iteration n°{}\n".format(acq_m, -best_observed_seed[-1],
                                                                                      best_n_iter))

    return best_observed_seed,  new_opt_x_list
