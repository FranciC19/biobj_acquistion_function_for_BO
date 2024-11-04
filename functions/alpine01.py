import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from typing import List, Optional, Tuple

from functions.function import Function
from utils.args_manager import DEVICE, TORCH_TYPE


class alpine01(SyntheticTestFunction):
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
        self._bounds = [(-10, 10) for _ in range(self.dim)]
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

    def evaluate_true(self, X: torch.Tensor):
        expr = torch.mul(X, torch.sin(X)) + torch.mul(X, 0.1)
        return torch.norm(expr, p=1, dim=-1)


class Alpine01(Function):

    def __init__(self, x_size: int):
        Function.__init__(self)

        self._obj_f = alpine01(dim=x_size, noise_std=None, negate=True, bounds=None)
        self._best_obj_f = 0
        self._bounds = torch.tensor(
            [[self._obj_f._bounds[d][0] for d in range(x_size)], [self._obj_f._bounds[d][1] for d in range(x_size)]],
            device=DEVICE, dtype=TORCH_TYPE)

    @staticmethod
    def name():
        return 'Alpine01'
