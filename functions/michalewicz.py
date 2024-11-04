import torch
from botorch.test_functions.synthetic import Michalewicz as michalewicz

from functions.function import Function
from utils.args_manager import DEVICE, TORCH_TYPE


class Michalewicz(Function):

    def __init__(self, x_size: int):
        assert x_size in [2, 5, 10]

        Function.__init__(self)

        self._obj_f = michalewicz(dim=x_size, noise_std=None, negate=True, bounds=None)
        optvals_dict = {2: 1.80130341, 5: 4.687658, 10: 9.66015}
        self._best_obj_f = optvals_dict[x_size]
        self._bounds = torch.tensor(
            [[self._obj_f._bounds[d][0] for d in range(x_size)], [self._obj_f._bounds[d][1] for d in range(x_size)]],
            device=DEVICE, dtype=TORCH_TYPE)

    @staticmethod
    def name():
        return 'Michalewicz'
