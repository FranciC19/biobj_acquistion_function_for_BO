import numpy as np
from abc import ABC
import torch


class Function(ABC):

    def __init__(self):
        self._obj_f = None
        self._best_obj_f = np.inf
        self._bounds = None

    def eval(self, x: torch.Tensor):
        return self._obj_f(x).unsqueeze(-1)

    def dim(self):
        return self._obj_f.dim

    @property
    def best_obj_f(self):
        return self._best_obj_f

    @property
    def bounds(self):
        return self._bounds

    @staticmethod
    def name():
        pass
