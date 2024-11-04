#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Batch acquisition functions using the reparameterization trick in combination
with (quasi) Monte-Carlo sampling. See [Rezende2014reparam]_, [Wilson2017reparam]_ and
[Balandat2020botorch]_.

.. [Rezende2014reparam]
    D. J. Rezende, S. Mohamed, and D. Wierstra. Stochastic backpropagation and
    approximate inference in deep generative models. ICML 2014.

.. [Wilson2017reparam]
    J. T. Wilson, R. Moriconi, F. Hutter, and M. P. Deisenroth.
    The reparameterization trick for acquisition functions. ArXiv 2017.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from typing import Any, Callable, List, Optional, Protocol, Tuple, Union

import torch
from botorch import acquisition
from botorch.acquisition.acquisition import AcquisitionFunction, MCSamplerMixin
from botorch.acquisition.cached_cholesky import CachedCholeskyMCAcquisitionFunction
from botorch.acquisition.objective import (
    ConstrainedMCObjective,
    IdentityMCObjective,
    MCAcquisitionObjective,
    PosteriorTransform,
)
from botorch.acquisition.utils import prune_inferior_points
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.utils.objective import compute_constraint_indicator
from botorch.utils.transforms import (
    concatenate_pending_points,
    match_batch_shape,
    t_batch_mode_transform,
)
from torch import Tensor


class MCAcquisitionFunction(AcquisitionFunction, MCSamplerMixin, ABC):
    r"""
    Abstract base class for Monte-Carlo based batch acquisition functions.

    :meta private:
    """

    def __init__(
        self,
        model: Model,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        r"""
        Args:
            model: A fitted model.
            sampler: The sampler used to draw base samples. If not given,
                a sampler is generated using `get_sampler`.
                NOTE: For posteriors that do not support base samples,
                a sampler compatible with intended use case must be provided.
                See `ForkedRNGSampler` and `StochasticSampler` as examples.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            posterior_transform: A PosteriorTransform (optional).
            X_pending: A `batch_shape, m x d`-dim Tensor of `m` design points
                that have points that have been submitted for function evaluation
                but have not yet been evaluated.
        """
        super().__init__(model=model)
        MCSamplerMixin.__init__(self, sampler=sampler)
        if objective is None and model.num_outputs != 1:
            if posterior_transform is None:
                raise UnsupportedError(
                    "Must specify an objective or a posterior transform when using "
                    "a multi-output model."
                )
            elif not posterior_transform.scalarize:
                raise UnsupportedError(
                    "If using a multi-output model without an objective, "
                    "posterior_transform must scalarize the output."
                )
        if objective is None:
            objective = IdentityMCObjective()
        self.posterior_transform = posterior_transform
        self.objective: MCAcquisitionObjective = objective
        self.set_X_pending(X_pending)

    def _get_samples_and_objectives(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes posterior samples and objective values at input X.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of model inputs.

        Returns:
            samples: `sample_shape x batch_shape x q x m`-dim posterior samples.
            obj: A `sample_shape x batch_shape x q`-dim Tensor of MC objective values.
        """
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        samples = self.get_posterior_samples(posterior)
        return samples, self.objective(samples=samples, X=X)

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        r"""Takes in a `batch_shape x q x d` X Tensor of t-batches with `q` `d`-dim
        design points each, and returns a Tensor with shape `batch_shape'`, where
        `batch_shape'` is the broadcasted batch shape of model and input `X`. Should
        utilize the result of `set_X_pending` as needed to account for pending function
        evaluations.
        """
        pass  # pragma: no cover


class SampleReducingMCAcquisitionFunction(MCAcquisitionFunction):
    r"""MC-based batch acquisition function that reduces across samples.

    This computes the - possibly constrained - acquisition value by
    (1) computing the unconstrained utility for each MC sample using `sample_forward`,
    (2) weighing the utility values by the constraint indicator per MC sample, and
    (3) reducing (e.g. averaging) the weighted utility values over the MC dimension.
    """

    def __init__(
        self,
        model: Model,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
        sample_reduction: SampleReductionProtocol = torch.mean,
        q_reduction: SampleReductionProtocol = torch.amax,
        constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
        eta: Union[Tensor, float] = 1e-3,
    ):
        r"""Constructor of SampleReducingMCAcquisitionFunction.

        Args:
            model: A fitted model.
            sampler: The sampler used to draw base samples. If not given,
                a sampler is generated using `get_sampler`.
                NOTE: For posteriors that do not support base samples,
                a sampler compatible with intended use case must be provided.
                See `ForkedRNGSampler` and `StochasticSampler` as examples.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            posterior_transform: A PosteriorTransform (optional).
            X_pending: A `batch_shape, m x d`-dim Tensor of `m` design points
                that have points that have been submitted for function evaluation
                but have not yet been evaluated.
            sample_reduction: A callable that takes in a `sample_shape x batch_shape`
                Tensor of acquisition utility values, a keyword-argument `dim` that
                specifies the sample dimensions to reduce over, and returns a
                `batch_shape`-dim Tensor of acquisition values.
            q_reduction: A callable that takes in a `sample_shape x batch_shape x q`
                Tensor of acquisition utility values, a keyword-argument `dim` that
                specifies the q dimension to reduce over (i.e. -1), and returns a
                `sample_shape x batch_shape`-dim Tensor of acquisition values.
            constraints: A list of constraint callables which map a Tensor of posterior
                samples of dimension `sample_shape x batch-shape x q x m`-dim to a
                `sample_shape x batch-shape x q`-dim Tensor. The associated constraints
                are considered satisfied if the output is less than zero.
                NOTE: Constraint-weighting is only compatible with non-negative
                acquistion utilities, e.g. all improvement-based acquisition functions.
            eta: Temperature parameter(s) governing the smoothness of the sigmoid
                approximation to the constraint indicators. See the docs of
                `compute_constraint_indicator` for more details on this parameter.
        """
        if constraints is not None and isinstance(objective, ConstrainedMCObjective):
            raise ValueError(
                "ConstrainedMCObjective as well as constraints passed to constructor."
                "Choose one or the other, preferably the latter."
            )
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        # Shall the need arise, sample_dim could be exposed in the constructor.
        sample_dim = tuple(range(len(self.sample_shape)))
        self._sample_reduction = partial(sample_reduction, dim=sample_dim)
        self._q_reduction = partial(q_reduction, dim=-1)
        self._constraints = constraints
        self._eta = eta

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Takes in a `batch_shape x q x d` Tensor of t-batches with `q` `d`-dim
        design points each, and returns a Tensor with shape `batch_shape'`, where
        `batch_shape'` is the broadcasted batch shape of model and input `X`. Should
        utilize the result of `set_X_pending` as needed to account for pending function
        evaluations.
        """
        samples, obj = self._get_samples_and_objectives(X)
        acqval = self._sample_forward(obj)  # `sample_sample x batch_shape x q`
        acqval = self._apply_constraints(acqval=acqval, samples=samples)
        return self._sample_reduction(self._q_reduction(acqval))

    @abstractmethod
    def _sample_forward(self, obj: Tensor) -> Tensor:
        """Evaluates the acquisition utility per MC sample based on objective value obj.

        Args:
            obj: A `sample_shape x batch_shape x q`-dim Tensor of MC objective values.

        Returns:
            A `sample_shape x batch_shape x q`-dim Tensor of acquisition utility values.
        """
        pass  # pragma: no cover

    def _apply_constraints(self, acqval: Tensor, samples: Tensor) -> Tensor:
        """Multiplies the acquisition utility by constraint indicators.

        Args:
            acqval: `sample_shape x batch_shape x q`-dim acquisition utility values.
            samples: `sample_shape x batch_shape x q x m`-dim posterior samples.

        Returns:
            A `sample_shape x batch_shape x q`-dim Tensor of acquisition utility values
                multiplied by a smoothed constraint indicator per sample.
        """
        if self._constraints is not None:
            if (acqval < 0).any():
                raise ValueError(
                    "Constraint-weighting requires unconstrained "
                    "acquisition values to be non-negative."
                )
            acqval = acqval * self._compute_constraint_indicator(samples)
        return acqval

    def _compute_constraint_indicator(self, samples: Tensor) -> Tensor:
        """Computes a sigmoid approximation to the constraint indicator per sample.

        Args:
            samples: `sample_shape x batch_shape x q x m`-dim posterior samples.

        Returns:
            A `samples_shape x batch_shape x q`-dim Tensor of constraint indicators.
        """
        return compute_constraint_indicator(  # convenient to also call from qNEI
            constraints=self._constraints, samples=samples, eta=self._eta
        )


class SampleReductionProtocol(Protocol):
    """For static type check of SampleReducingMCAcquisitionFunction's mc_reduction."""

    @staticmethod
    def __call__(X: Tensor, *, dim: torch.Size) -> Tensor:
        pass  # pragma: no cover


class qExpectedImprovement(SampleReducingMCAcquisitionFunction):
    r"""MC-based batch Expected Improvement.

    This computes qEI by
    (1) sampling the joint posterior over q points
    (2) evaluating the improvement over the current best for each sample
    (3) maximizing over q
    (4) averaging over the samples

    `qEI(X) = E(max(max Y - best_f, 0)), Y ~ f(X), where X = (x_1,...,x_q)`

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> best_f = train_Y.max()[0]
        >>> sampler = SobolQMCNormalSampler(1024)
        >>> qEI = qExpectedImprovement(model, best_f, sampler)
        >>> qei = qEI(test_X)
    """

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
        constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
        eta: Union[Tensor, float] = 1e-3,
        **kwargs: Any,
    ) -> None:
        r"""q-Expected Improvement.

        Args:
            model: A fitted model.
            best_f: The best objective value observed so far (assumed noiseless). Can be
                a `batch_shape`-shaped tensor, which in case of a batched model
                specifies potentially different values for each element of the batch.
            sampler: The sampler used to draw base samples. See `MCAcquisitionFunction`
                more details.
            objective: The MCAcquisitionObjective under which the samples are evaluated.
                Defaults to `IdentityMCObjective()`.
            posterior_transform: A PosteriorTransform (optional).
            X_pending:  A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation but have not yet been evaluated.
                Concatenated into X upon forward call. Copied and set to have no
                gradient.
            constraints: A list of constraint callables which map posterior samples to
                a scalar. The associated constraint is considered satisfied if this
                scalar is less than zero.
            eta: Temperature parameter(s) governing the smoothness of the sigmoid
                approximation to the constraint indicators. See the docs of
                `compute_constraint_indicator` for more details on this parameter.
        """
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            constraints=constraints,
            eta=eta,
        )
        self.register_buffer("best_f", torch.as_tensor(best_f, dtype=float))

    def _sample_forward(self, obj: Tensor) -> Tensor:
        r"""Evaluate qExpectedImprovement per sample on the candidate set `X`.

        Args:
            obj: A `sample_shape x batch_shape x q`-dim Tensor of MC objective values.

        Returns:
            A `sample_shape x batch_shape x q`-dim Tensor of improvement utility values.
        """
        return (obj - self.best_f.unsqueeze(-1).to(obj)).clamp_min(0)


class qNoisyExpectedImprovement(
    SampleReducingMCAcquisitionFunction, CachedCholeskyMCAcquisitionFunction
):
    r"""MC-based batch Noisy Expected Improvement.

    This function does not assume a `best_f` is known (which would require
    noiseless observations). Instead, it uses samples from the joint posterior
    over the `q` test points and previously observed points. The improvement
    over previously observed points is computed for each sample and averaged.

    `qNEI(X) = E(max(max Y - max Y_baseline, 0))`, where
    `(Y, Y_baseline) ~ f((X, X_baseline)), X = (x_1,...,x_q)`

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> sampler = SobolQMCNormalSampler(1024)
        >>> qNEI = qNoisyExpectedImprovement(model, train_X, sampler)
        >>> qnei = qNEI(test_X)
    """

    def __init__(
        self,
        model: Model,
        X_baseline: Tensor,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
        prune_baseline: bool = True,
        cache_root: bool = True,
        constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
        eta: Union[Tensor, float] = 1e-3,
        **kwargs: Any,
    ) -> None:
        r"""q-Noisy Expected Improvement.

        Args:
            model: A fitted model.
            X_baseline: A `batch_shape x r x d`-dim Tensor of `r` design points
                that have already been observed. These points are considered as
                the potential best design point.
            sampler: The sampler used to draw base samples. See `MCAcquisitionFunction`
                more details.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            posterior_transform: A PosteriorTransform (optional).
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points
                that have points that have been submitted for function evaluation
                but have not yet been evaluated. Concatenated into `X` upon
                forward call. Copied and set to have no gradient.
            prune_baseline: If True, remove points in `X_baseline` that are
                highly unlikely to be the best point. This can significantly
                improve performance and is generally recommended. In order to
                customize pruning parameters, instead manually call
                `botorch.acquisition.utils.prune_inferior_points` on `X_baseline`
                before instantiating the acquisition function.
            cache_root: A boolean indicating whether to cache the root
                decomposition over `X_baseline` and use low-rank updates.
            constraints: A list of constraint callables which map posterior samples to
                a scalar. The associated constraint is considered satisfied if this
                scalar is less than zero.
            eta: Temperature parameter(s) governing the smoothness of the sigmoid
                approximation to the constraint indicators. See the docs of
                `compute_constraint_indicator` for more details on this parameter.

        TODO: similar to qNEHVI, when we are using sequential greedy candidate
        selection, we could incorporate pending points X_baseline and compute
        the incremental qNEI from the new point. This would greatly increase
        efficiency for large batches.
        """
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            constraints=constraints,
            eta=eta,
        )
        self._setup(model=model, cache_root=cache_root)
        if prune_baseline:
            X_baseline = prune_inferior_points(
                model=model,
                X=X_baseline,
                objective=objective,
                posterior_transform=posterior_transform,
                marginalize_dim=kwargs.get("marginalize_dim"),
            )
        self.register_buffer("X_baseline", X_baseline)
        # registering buffers for _get_samples_and_objectives or in the next `if` block
        self.register_buffer("baseline_samples", None)
        self.register_buffer("baseline_obj", None)
        if self._cache_root:
            self.q_in = -1
            # set baseline samples
            with torch.no_grad():  # this is _get_samples_and_objectives(X_baseline)
                posterior = self.model.posterior(
                    X_baseline, posterior_transform=self.posterior_transform
                )
                # Note: The root decomposition is cached in two different places. It
                # may be confusing to have two different caches, but this is not
                # trivial to change since each is needed for a different reason:
                # - LinearOperator caching to `posterior.mvn` allows for reuse within
                #  this function, which may be helpful if the same root decomposition
                #  is produced by the calls to `self.base_sampler` and
                #  `self._cache_root_decomposition`.
                # - self._baseline_L allows a root decomposition to be persisted outside
                #   this method.
                baseline_samples = self.get_posterior_samples(posterior)
                baseline_obj = self.objective(baseline_samples, X=X_baseline)

            # We make a copy here because we will write an attribute `base_samples`
            # to `self.base_sampler.base_samples`, and we don't want to mutate
            # `self.sampler`.
            self.base_sampler = deepcopy(self.sampler)
            self.baseline_samples = baseline_samples
            self.baseline_obj = baseline_obj
            self.register_buffer(
                "_baseline_best_f",
                self._compute_best_feasible_objective(
                    samples=baseline_samples, obj=baseline_obj
                ),
            )
            self._baseline_L = self._compute_root_decomposition(posterior=posterior)

    def best_f(self, obj: Tensor) -> Tensor:
        """Computes the best (feasible) noisy objective value.

        Args:
            obj: `sample_shape x batch_shape x q`-dim Tensor of objectives in forward.

        Returns:
            `sample_shape x batch_shape x 1`-dim Tensor of best feasible objectives.
        """
        if self._cache_root:
            val = self._baseline_best_f
        else:
            val = self._compute_best_feasible_objective(
                samples=self.baseline_samples, obj=self.baseline_obj
            )
        # ensuring shape, dtype, device compatability with obj
        n_sample_dims = len(self.sample_shape)
        view_shape = torch.Size(
            [
                *val.shape[:n_sample_dims],  # sample dimensions
                *(1,) * (obj.ndim - val.ndim),  # pad to match obj
                *val.shape[n_sample_dims:],  # the rest
            ]
        )
        return val.view(view_shape).to(obj)

    def _sample_forward(self, obj: Tensor) -> Tensor:
        r"""Evaluate qNoisyExpectedImprovement per sample on the candidate set `X`.

        Args:
            obj: A `sample_shape x batch_shape x q`-dim Tensor of MC objective values.

        Returns:
            A `sample_shape x batch_shape x q`-dim Tensor of noisy improvement values.
        """
        return (obj - self.best_f(obj)).clamp_min(0)

    def _get_samples_and_objectives(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Compute samples at new points, using the cached root decomposition.

        Args:
            X: A `batch_shape x q x d`-dim tensor of inputs.

        Returns:
            samples: `sample_shape x batch_shape x q x m`-dim posterior samples.
            obj: A `sample_shape x batch_shape x q`-dim Tensor of MC objective values.
        """
        q = X.shape[-2]
        X_full = torch.cat([match_batch_shape(self.X_baseline, X), X], dim=-2)
        # TODO: Implement more efficient way to compute posterior over both training and
        # test points in GPyTorch (https://github.com/cornellius-gp/gpytorch/issues/567)
        posterior = self.model.posterior(
            X_full, posterior_transform=self.posterior_transform
        )
        if not self._cache_root:
            samples_full = super().get_posterior_samples(posterior)
            samples = samples_full[..., -q:, :]
            obj_full = self.objective(samples_full, X=X_full)
            # assigning baseline buffers so `best_f` can be computed in _sample_forward
            self.baseline_obj, obj = obj_full[..., :-q], obj_full[..., -q:]
            self.baseline_samples = samples_full[..., :-q, :]
            return samples, obj

        # handle one-to-many input transforms
        n_plus_q = X_full.shape[-2]
        n_w = posterior._extended_shape()[-2] // n_plus_q
        q_in = q * n_w
        self._set_sampler(q_in=q_in, posterior=posterior)
        samples = self._get_f_X_samples(posterior=posterior, q_in=q_in)
        obj = self.objective(samples, X=X_full[..., -q:, :])
        return samples, obj

    def _compute_best_feasible_objective(self, samples: Tensor, obj: Tensor) -> Tensor:
        """
        Args:
            samples: `sample_shape x batch_shape x q x m`-dim posterior samples.
            obj: A `sample_shape x batch_shape x q`-dim Tensor of MC objective values.

        Returns:
            A `sample_shape x batch_shape x 1`-dim Tensor of best feasible objectives.
        """
        if self._constraints is not None:
            # is_feasible is sample_shape x batch_shape x q
            is_feasible = self._compute_constraint_indicator(samples)
            is_feasible = is_feasible > 0.5  # due to sigmoid approximation
            if is_feasible.any():
                obj = torch.where(is_feasible, obj, -torch.inf)
            else:  # if there are no feasible observations, estimate a lower
                # bound on the objective by sampling convex combinations of X_baseline.
                convex_weights = torch.rand(
                    32,
                    self.X_baseline.shape[-2],
                    dtype=self.X_baseline.dtype,
                    device=self.X_baseline.device,
                )
                weights_sum = convex_weights.sum(dim=0, keepdim=True)
                convex_weights = convex_weights / weights_sum
                # infeasible cost M is such that -M < min_x f(x), thus
                # 0 < min_x f(x) - (-M), so we should take -M as a lower
                # bound on the best feasible objective
                return -acquisition.utils.get_infeasible_cost(
                    X=convex_weights @ self.X_baseline,
                    model=self.model,
                    objective=self.objective,
                    posterior_transform=self.posterior_transform,
                ).expand(*obj.shape[:-1], 1)

        # we don't need to differentiate through X_baseline for now, so taking
        # the regular max over the n points to get best_f is fine
        with torch.no_grad():
            return obj.max(dim=-1, keepdim=True).values


class qProbabilityOfImprovement(SampleReducingMCAcquisitionFunction):
    r"""MC-based batch Probability of Improvement.

    Estimates the probability of improvement over the current best observed
    value by sampling from the joint posterior distribution of the q-batch.
    MC-based estimates of a probability involves taking expectation of an
    indicator function; to support auto-differntiation, the indicator is
    replaced with a sigmoid function with temperature parameter `tau`.

    `qPI(X) = P(max Y >= best_f), Y ~ f(X), X = (x_1,...,x_q)`

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> best_f = train_Y.max()[0]
        >>> sampler = SobolQMCNormalSampler(1024)
        >>> qPI = qProbabilityOfImprovement(model, best_f, sampler)
        >>> qpi = qPI(test_X)
    """

    def __init__(
        self,
        model: Model,
        best_f: Union[float, Tensor],
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
        tau: float = 1e-3,
        constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
        eta: Union[Tensor, float] = 1e-3,
    ) -> None:
        r"""q-Probability of Improvement.

        Args:
            model: A fitted model.
            best_f: The best objective value observed so far (assumed noiseless). Can
                be a `batch_shape`-shaped tensor, which in case of a batched model
                specifies potentially different values for each element of the batch.
            sampler: The sampler used to draw base samples. See `MCAcquisitionFunction`
                more details.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            posterior_transform: A PosteriorTransform (optional).
            X_pending:  A `m x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation
                but have not yet been evaluated.  Concatenated into X upon
                forward call.  Copied and set to have no gradient.
            tau: The temperature parameter used in the sigmoid approximation
                of the step function. Smaller values yield more accurate
                approximations of the function, but result in gradients
                estimates with higher variance.
            constraints: A list of constraint callables which map posterior samples to
                a scalar. The associated constraint is considered satisfied if this
                scalar is less than zero.
            eta: Temperature parameter(s) governing the smoothness of the sigmoid
                approximation to the constraint indicators. See the docs of
                `compute_constraint_indicator` for more details on this parameter.
        """
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            constraints=constraints,
            eta=eta,
        )
        best_f = torch.as_tensor(best_f, dtype=float).unsqueeze(-1)  # adding batch dim
        self.register_buffer("best_f", best_f)
        self.register_buffer("tau", torch.as_tensor(tau, dtype=float))

    def _sample_forward(self, obj: Tensor) -> Tensor:
        r"""Evaluate qProbabilityOfImprovement per sample on the candidate set `X`.

        Args:
            obj: A `sample_shape x batch_shape x q`-dim Tensor of MC objective values.

        Returns:
            A `sample_shape x batch_shape x q`-dim Tensor of improvement indicators.
        """
        improvement = obj - self.best_f.to(obj)
        return torch.sigmoid(improvement / self.tau)


class qSimpleRegret(SampleReducingMCAcquisitionFunction):
    r"""MC-based batch Simple Regret.

    Samples from the joint posterior over the q-batch and computes the simple regret.

    `qSR(X) = E(max Y), Y ~ f(X), X = (x_1,...,x_q)`

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> sampler = SobolQMCNormalSampler(1024)
        >>> qSR = qSimpleRegret(model, sampler)
        >>> qsr = qSR(test_X)
    """

    def _sample_forward(self, obj: Tensor) -> Tensor:
        r"""Evaluate qSimpleRegret per sample on the candidate set `X`.

        Args:
            obj: A `sample_shape x batch_shape x q`-dim Tensor of MC objective values.

        Returns:
            A `sample_shape x batch_shape x q`-dim Tensor of simple regret values.
        """
        return obj


class qUpperConfidenceBound(SampleReducingMCAcquisitionFunction):
    r"""MC-based batch Upper Confidence Bound.

    Uses a reparameterization to extend UCB to qUCB for q > 1 (See Appendix A
    of [Wilson2017reparam].)

    `qUCB = E(max(mu + |Y_tilde - mu|))`, where `Y_tilde ~ N(mu, beta pi/2 Sigma)`
    and `f(X)` has distribution `N(mu, Sigma)`.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> sampler = SobolQMCNormalSampler(1024)
        >>> qUCB = qUpperConfidenceBound(model, 0.1, sampler)
        >>> qucb = qUCB(test_X)
    """

    def __init__(
        self,
        model: Model,
        beta: float,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        r"""q-Upper Confidence Bound.

        Args:
            model: A fitted model.
            beta: Controls tradeoff between mean and standard deviation in UCB.
            sampler: The sampler used to draw base samples. See `MCAcquisitionFunction`
                more details.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            posterior_transform: A PosteriorTransform (optional).
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation but have not yet
                been evaluated. Concatenated into X upon forward call. Copied and set to
                have no gradient.
        """
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        self.beta_prime = math.sqrt(beta * math.pi / 2)

    def _sample_forward(self, obj: Tensor) -> Tensor:
        r"""Evaluate qUpperConfidenceBound per sample on the candidate set `X`.

        Args:
            obj: A `sample_shape x batch_shape x q`-dim Tensor of MC objective values.

        Returns:
            A `sample_shape x batch_shape x q`-dim Tensor of acquisition values.
        """
        mean = obj.mean(dim=0)
        return mean + self.beta_prime * (obj - mean).abs()
