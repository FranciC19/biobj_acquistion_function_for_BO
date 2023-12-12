import torch
from botorch.test_functions.synthetic import EggHolder as eggholder

from functions.function import Function
from utils.args_manager import DEVICE, TORCH_TYPE


class EggHolder(Function):

    def __init__(self, x_size):
        assert x_size == 2

        Function.__init__(self)

        self._obj_f = eggholder(noise_std=None, negate=True, bounds=None)
        self._best_obj_f = 959.6407
        self._bounds = torch.tensor(
            [[self._obj_f._bounds[d][0] for d in range(x_size)], [self._obj_f._bounds[d][1] for d in range(x_size)]],
            device=DEVICE, dtype=TORCH_TYPE)

    @staticmethod
    def name():
        return 'EggHolder'
