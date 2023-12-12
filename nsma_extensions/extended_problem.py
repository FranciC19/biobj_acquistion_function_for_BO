from abc import ABC
import numpy as np
from nsma.problems.problem import Problem


class ExtendedProblem(ABC, Problem):

    def __init__(self, n: int):
        Problem.__init__(self, n)

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
