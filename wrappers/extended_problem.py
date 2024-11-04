from abc import ABC, abstractmethod
import numpy as np
import torch

from nsma.problems.problem import Problem


class ExtendedProblem(ABC, Problem):

    def __init__(self, n: int):
        Problem.__init__(self, n)

    @abstractmethod
    def evaluate_functions_tensor(self, x: torch.Tensor):
        pass

    @property
    def objectives(self):
        raise RuntimeError

    @objectives.setter
    def objectives(self, objectives: list):
        for obj in objectives:
            assert obj is not np.nan and obj is not np.inf and obj is not -np.inf
        self.__objectives = objectives

    @property
    def m(self):
        return len(self.__objectives)
