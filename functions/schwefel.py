import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from typing import List, Optional, Tuple

from functions.function import Function
from utils.args_manager import DEVICE, TORCH_TYPE


class schwefel(SyntheticTestFunction):
    _optimal_value = 0.0
    _check_grad_at_opt: bool = False

    def __init__(
            self,
            dim: int = 2,
            noise_std: Optional[float] = None,
            negate: bool = False,
            bounds: Optional[List[Tuple[float, float]]] = None,
    ):
        self.dim = dim
        self._bounds = [(-500, 500) for _ in range(self.dim)]
        self._optimizers = [tuple(420.9687 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

    def evaluate_true(self, X: torch.Tensor):
        expr_1 = torch.mul(418.9829, self.dim)
        expr_2 = torch.mul(X, torch.sin(torch.sqrt(torch.abs(X))))
        return expr_1 - torch.sum(expr_2, dim=1)


class Schwefel(Function):

    def __init__(self, x_size: int):
        Function.__init__(self)

        self._obj_f = schwefel(dim=x_size, noise_std=None, negate=True, bounds=None)
        self._best_obj_f = 0
        self._bounds = torch.tensor(
            [[self._obj_f._bounds[d][0] for d in range(x_size)], [self._obj_f._bounds[d][1] for d in range(x_size)]],
            device=DEVICE, dtype=TORCH_TYPE)

    @staticmethod
    def name():
        return 'Schwefel'
