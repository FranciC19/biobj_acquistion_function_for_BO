import torch
from botorch.test_functions.synthetic import Shekel as shekel

from functions.function import Function
from utils.args_manager import DEVICE, TORCH_TYPE


class Shekel(Function):

    def __init__(self, x_size: int):
        assert x_size == 4

        Function.__init__(self)

        # m=10 if not specified
        self._obj_f = shekel(m=10, noise_std=None, negate=True, bounds=None)
        self._best_obj_f = 10.5363
        self._bounds = torch.tensor(
            [[self._obj_f._bounds[d][0] for d in range(x_size)], [self._obj_f._bounds[d][1] for d in range(x_size)]],
            device=DEVICE, dtype=TORCH_TYPE)

    @staticmethod
    def name():
        return 'Shekel'
