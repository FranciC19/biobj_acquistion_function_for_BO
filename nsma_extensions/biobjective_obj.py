import numpy as np
import torch
from torch.autograd.functional import jacobian
from nsma_extensions.extended_problem import ExtendedProblem

from utils.args_manager import DEVICE, TORCH_TYPE


class BiObjectiveProblem(ExtendedProblem):
    def __init__(self, dim: int, model, lb, ub):
        ExtendedProblem.__init__(self, dim)

        self.__model_posterior = model.posterior

        self.objectives = [self.mu_x, self.var_x]
        # NSMA/NSGA bounds 
        self.lb = lb
        self.ub = ub

    def mu_x(self, x0):
        return -self.__model_posterior(x0).mvn.mean

    def var_x(self, x0):
        return -self.__model_posterior(x0).mvn.variance

    def evaluate_functions(self, x: np.array):
        x = torch.tensor(x.reshape((1, self.n)), device=DEVICE, dtype=TORCH_TYPE)
        mp = self.__model_posterior(x)
        return np.array([-mp.mean.numpy(force=True)[0][0], -mp.variance.numpy(force=True)[0][0]])

    def evaluate_functions_tensor(self, x: torch.Tensor):
        mp = self.__model_posterior(x)
        return (-mp.mean[0], -mp.variance[0])

    def evaluate_functions_jacobian(self, x: np.array):
        x_tensor_reshaped = torch.tensor(x.reshape((1, self.n)), device=DEVICE, dtype=TORCH_TYPE, requires_grad=True)
        jacobian_val = jacobian(self.evaluate_functions_tensor, x_tensor_reshaped)

        return np.array([jacobian_val[0][0, 0].numpy(force=True), jacobian_val[1][0, 0].numpy(force=True)])

    @staticmethod
    def name():
        return 'BiObjectiveProblem'

    @staticmethod
    def family_name():
        return 'BiObjectiveProblem'
