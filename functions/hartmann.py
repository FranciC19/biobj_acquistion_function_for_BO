import torch
from botorch.test_functions.synthetic import Hartmann as hartmann

from functions.function import Function
from utils.args_manager import DEVICE, TORCH_TYPE


class Hartmann(Function):

    def __init__(self, x_size: int):
        assert x_size in [3, 6]

        Function.__init__(self)

        self._obj_f = hartmann(dim=x_size, noise_std=None, negate=True, bounds=None)
        self._best_obj_f = 3.32237 if x_size == 6 else 3.86278
        self._bounds = torch.tensor(
            [[self._obj_f._bounds[d][0] for d in range(x_size)], [self._obj_f._bounds[d][1] for d in range(x_size)]],
            device=DEVICE, dtype=TORCH_TYPE)

    @staticmethod
    def name():
        return 'Hartmann'
