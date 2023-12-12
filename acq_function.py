import torch
import numpy as np
from botorch.acquisition.monte_carlo import qExpectedImprovement, qUpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.utils.transforms import unnormalize


class AcqFunction:

    def __init__(self, acq_name, model, fun, train_obj, qmc_sampler, ucb_beta=None, normalized_fun=True):

        self.__acq_name = acq_name

        self.normalized_fun = normalized_fun

        if acq_name == 'qei':
            self.__acq_f = qExpectedImprovement(model=model, best_f=train_obj.max(), sampler=qmc_sampler)

        elif acq_name == 'qucb':
            self.__acq_f = qUpperConfidenceBound(model=model, beta=ucb_beta, sampler=qmc_sampler)

        else:
            raise NotImplementedError('Acquisition function undefined!')

        self._fun = fun

        self.bounds = torch.tensor([np.zeros((fun.dim())), np.ones((fun.dim()))])

    def optimize(self, train_x, batch_size):

        try:
            candidates, _ = optimize_acqf(
                acq_function=self.__acq_f,
                bounds=self.bounds,
                q=batch_size,
                num_restarts=train_x.size(dim=0),
                options={"maxiter": 100},
                batch_initial_conditions=train_x
            )
        except:
            try:
                candidates, _ = optimize_acqf(
                    acq_function=self.__acq_f,
                    bounds=self.bounds,
                    q=batch_size,
                    num_restarts=train_x.size(dim=0),
                    options={"maxiter": 100},
                    raw_samples=train_x.size(dim=0)
                )
            except:
                return None, None

        new_opt_x = candidates.detach()
        new_opt_x = unnormalize(new_opt_x, self._fun._bounds)
        new_opt_obj = self._fun.eval(new_opt_x)

        return new_opt_x, new_opt_obj
