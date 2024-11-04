from __future__ import annotations

import math

from functools import partial

from typing import Callable, List, Optional, Union

import torch
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.acquisition.monte_carlo import SampleReducingMCAcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective, PosteriorTransform
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.utils.safe_math import logmeanexp
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from torch import Tensor
from torch.nn.functional import softplus

TAU_RELU = 1e-3
TAU_MAX = 1e-1
STANDARDIZE_TAU = False


class LogImprovementMCAcquisitionFunction(SampleReducingMCAcquisitionFunction):
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
        constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
        eta: Union[Tensor, float] = 1e-3,
        tau_max: float = TAU_MAX,
        log: bool = True,
        jensen_lower_bound: bool = False,
        q_sauce: bool = False,
        standardize_tau_max: bool = STANDARDIZE_TAU,
        fatten: bool = True,
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
            sample_reduction: A SampleReductionProtocol, i.e. a callable that takes in
                a `sample_shape x batch_shape` Tensor of acquisition utility values, a
                keyword-argument `dim` that specifies the sample dimensions to reduce
                over, and returns a `batch_shape`-dim Tensor of acquisition values.
            constraints: A list of constraint callables which map posterior samples to
                a scalar. The associated constraint is considered satisfied if this
                scalar is less than zero.
            eta: Temperature parameter(s) governing the smoothness of the sigmoid
                approximation to the constraint indicators. See the docs of
                `compute_(log_)constraint_indicator` for more details on this parameter.
            tau_max: Temperature parameter controlling the sharpness of the
                approximation to the `max` operator over the `q` candidate points.
            log: If True, return the log smoothed EI. Otherwise, exponentiate before
                returning, primarily useful for ablation studies. Defaults to True.
            q_sauce: Switching order of maximum and relu approximations (experimental).
            standardize_tau_max: Standardizes the improvement of each candidate of a q batch,
                by the sample standard deviation before computing the acquisition
                utility. Important for asymptotic convergence.
            fatten: Toggles the logarithmic / linear asymptotic behavior of the smooth
                approximation to the ReLU.
        """
        q_reduction = partial(fatmax, tau=tau_max, standardize=standardize_tau_max)
        sample_reduction = torch.mean if jensen_lower_bound else logmeanexp
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            sample_reduction=sample_reduction,
            q_reduction=q_reduction,
            constraints=constraints,
            eta=eta,
        )
        self.tau_max = tau_max
        self.log = log
        self._q_sauce = q_sauce
        # self.register_buffer("best_f", torch.as_tensor(best_f, dtype=float))
        self.fatten = fatten  # passing this here so we can fatten the constraints too.

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Takes in a `batch_shape x q x d` Tensor of t-batches with `q` `d`-dim
        design points each, and returns a Tensor with shape `batch_shape'`, where
        `batch_shape'` is the broadcasted batch shape of model and input `X`. Should
        utilize the result of `set_X_pending` as needed to account for pending function
        evaluations.

        NOTE: We could overwrite the _apply_constraints method, and wouldn't need to
        redefine the forward call here if we're also fine getting rid of the log flag,
        which is seful for ablation studies, and include the q_sauce for non-log EI.
        """
        samples, obj = self._get_samples_and_objectives(X)
        # we can either get rid of this, or implement a fat q_reduction and see if it's
        # more efficient.
        if self._q_sauce and self._constraints is None:
            # reduce over obj first for efficiency, since sample_forward is monotonic
            # in obj, this is identical if q_reduction = max
            obj = self._q_reduction(obj)  # `sample_sample x batch_shape`
            acqval = self._sample_forward(obj.unsqueeze(-1)).squeeze(-1)
            acqval = self._sample_reduction(acqval)
        else:
            acqval = self._sample_forward(obj)  # `sample_sample x batch_shape x q`
            if self._constraints is not None:  # acqval assumed to be in log space.
                acqval = acqval + self._compute_log_constraint_indicator(samples)
            acqval = self._sample_reduction(self._q_reduction(acqval))
        return acqval if self.log else acqval.exp()

    def _compute_log_constraint_indicator(self, samples: Tensor) -> Tensor:
        """Computes the log of the sigmoid approximation to the constraint indicator.

        Args:
            samples: `sample_shape x batch_shape x q x m`-dim posterior

        Returns:
            A `samples_shape x batch_shape x q`-dim Tensor of constraint indicators.
        """
        return compute_log_constraint_indicator(  # convenient to call from qLogNEI
            constraints=self._constraints,
            samples=samples,
            eta=self._eta,
            fatten=self.fatten,
        )


class qLogExpectedImprovement(LogImprovementMCAcquisitionFunction):
    r"""MC-based batch Log Expected Smoothed Improvement.

    This computes qLogEI by
    (1) sampling the joint posterior over q points
    (2) evaluating the smoothed log improvement over the current best for each sample
    (3) smoothly maximizing over q
    (4) averaging over the samples in log space

    `qLogEI(X) ~ log(qEI(X)) = log(E(max(max Y - best_f, 0))),

    where Y ~ f(X), and X = (x_1,...,x_q)`.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> best_f = train_Y.max()[0]
        >>> sampler = SobolQMCNormalSampler(1024)
        >>> qLogEI = qLogExpectedImprovement(model, best_f, sampler)
        >>> qei = qLogEI(test_X)
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
        # these two are specific to LogImprovementMCAcquisitionFunctions
        log: bool = True,
        tau_max: float = TAU_MAX,
        jensen_lower_bound: bool = False,
        q_sauce: bool = False,
        standardize_tau_max: bool = STANDARDIZE_TAU,
        fatten: bool = True,
        # the below are specific to qLogEI
        standardize_tau_relu: bool = STANDARDIZE_TAU,
        tau_relu: float = TAU_RELU,
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
            tau_relu: Temperature parameter controlling the sharpness of the smooth
                approximations to ReLU. Default: 1e-2.
            tau_max: Temperature parameter controlling the sharpness of the smooth
                approximations to max. Default: 1e-2.
            jensen_lower_bound: if True, computes the expectation of the
                logarithm of the smoothed improvement, rather than the logarithm of the
                expected smoothed improvement. Due to Jensen's inequality, the expected
                log smoothed improvement is a lower bound on the log expected smoothed
                improvement. Defaults to False.
            log: If False, returns the non-log smoothed EI. Added for ablation studies.
                Defaults to True.
            quadratic_decay: Toggles the linear / quadratic asymptotic behavior of the
                smooth approximation to the improvement. Defaults to True.
            standardize: Standardizes the improvement of each candidate of a q batch,
                by the sample standard deviation. Important for asymptotic convergence.
            fatten: Toggles the logarithmic / linear asymptotic behavior of the smooth
                approximation to the ReLU.
        """
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            constraints=constraints,
            eta=eta,
            tau_max=check_tau(tau_max),
            log=log,
            jensen_lower_bound=jensen_lower_bound,
            q_sauce=q_sauce,
            standardize_tau_max=standardize_tau_max,
            fatten=fatten,
        )
        self.register_buffer("best_f", torch.as_tensor(best_f, dtype=float))
        self.tau_relu = check_tau(tau_relu)
        self.standardize_tau_relu = standardize_tau_relu

    def _sample_forward(self, obj: Tensor) -> Tensor:
        r"""Evaluate qLogExpectedImprovement on the candidate set `X`.

        Args:
            obj: `mc_shape x batch_shape x q`-dim Tensor of MC objective values.

        Returns:
            A `mc_shape x batch_shape x q`-dim Tensor of expected improvement values.
        """
        return _log_improvement(
            Y=obj,
            best_f=self.best_f,
            tau=self.tau_relu,
            standardize=self.standardize_tau_relu,
            fatten=self.fatten,
        )


class qLogProbabilityOfImprovement(LogImprovementMCAcquisitionFunction):
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
        >>> qPI = qLogProbabilityOfImprovement(model, best_f, sampler)
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
        constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
        eta: Union[Tensor, float] = 1e-3,
        # these two are specific to LogImprovementMCAcquisitionFunctions
        tau_max: float = TAU_MAX,
        standardize_tau_max: bool = STANDARDIZE_TAU,
        log: bool = True,
        jensen_lower_bound: bool = False,
        q_sauce: bool = False,
        # the below are specific to qLogPI
        tau_sigmoid: float = TAU_RELU,
        standardize_tau_sigmoid: bool = STANDARDIZE_TAU,
        fatten: bool = True,
    ) -> None:
        r"""q-LogProbability of Improvement.

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
        """
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            constraints=constraints,
            eta=eta,
            tau_max=tau_max,
            log=log,
            jensen_lower_bound=jensen_lower_bound,
            q_sauce=q_sauce,
            standardize_tau_max=standardize_tau_max,
            fatten=fatten,
        )
        best_f = torch.as_tensor(best_f, dtype=float).unsqueeze(-1)
        self.register_buffer("best_f", best_f)
        self.register_buffer("tau_sigmoid", torch.as_tensor(tau_sigmoid, dtype=float))
        self.standardize_tau_sigmoid

    def _sample_forward(self, obj: Tensor) -> Tensor:
        r"""Evaluate qLogProbabilityOfImprovement per sample on the candidate set `X`.

        Args:
            samples: `mc_shape x batch_shape x q x m`-dim Tensor of posterior samples.
            obj: `mc_shape x batch_shape x q`-dim Tensor of MC objective values.

        Returns:
            A `mc_shape x batch_shape x q`-dim Tensor of improvement indicator values.
        """
        log_sigmoid = log_fatmoid if self.fatten else logexpit
        tau = self.tau_sigmoid
        if self.standardize_tau_sigmoid:
            tau = tau * obj.std(dim=0, keepdim=True)
        return log_sigmoid((obj - self.best_f.to(obj)) / tau)


###################################### utils ##########################################
def _log_improvement(
    Y: Tensor,
    best_f: Tensor,
    tau: Union[float, Tensor],
    standardize: bool,
    fatten: bool,
) -> Tensor:
    """Computes the logarithm of the softplus-smoothed improvement, i.e.
        log_softplus(Y - best_f, beta=(1 / tau)).
    Note that softplus is an approximation to the regular ReLU objective whose maximum
    pointwise approximation error is linear with respect to tau as tau goes to zero.

    Args:
        obj: `mc_samples x batch_shape x q`-dim Tensor of output samples.
        best_f: Best previously observed objective value(s), broadcastable with obj.
        tau: Temperature parameter for smooth approximation of ReLU.
            as tau -> 0, maximum pointwise approximation error is linear w.r.t. tau.
        standardize: Toggles the temperature standardization of the smoothed function.
        fatten: Toggles the logarithmic / linear asymptotic behavior of the
            smooth approximation to ReLU.

    Returns:
        A `mc_samples x batch_shape x q`-dim Tensor of improvement values.
    """
    log_soft_clamp = log_fatplus if fatten else log_softplus
    Z = Y - best_f.to(Y)
    if standardize:  # scaling tau by standard deviation over mc dimension
        tau = tau * Y.std(dim=0, keepdim=True).clamp_max(1)
    return log_soft_clamp(Z, tau=tau)  # ~ ((Y - best_f) / Y_std).clamp(0)


def log_softplus(x: Tensor, tau: Union[float, Tensor] = 1.0) -> Tensor:
    """Computes the logarithm of the softplus function with high numerical accuracy.

    Args:
        x: Input tensor, should have single or double precision floats.
        tau: Decreasing tau increases the tightness of the
            approximation to ReLU. Non-negative and defaults to 1.0.

    Returns:
        Tensor corresponding to log(softplus(x)).
    """
    check_dtype_float32_or_float64(x)
    tau = torch.as_tensor(tau, dtype=x.dtype, device=x.device)
    # cutoff chosen to achieve accuracy to machine epsilon
    upper = 16 if x.dtype == torch.float32 else 32
    lower = -15 if x.dtype == torch.float32 else -35
    mask = x / tau > lower
    return torch.where(
        mask,
        softplus(x.masked_fill(~mask, lower), beta=(1 / tau), threshold=upper).log(),
        x / tau + tau.log(),
    )


def log_fatplus(x: Tensor, tau: Union[float, Tensor] = 1.0) -> Tensor:
    return fatplus(x, tau=tau).log()


def fatplus(x: Tensor, tau: Union[float, Tensor] = 1.0) -> Tensor:
    return tau * _fatplus(x / tau)


def _fatplus(x: Tensor) -> Tensor:
    alpha = 1e-1  # guarantees monotonicity and convexity (see Lemma 4)
    return softplus(x) + alpha * cauchy(x)


def fatmax(
    x: Tensor, dim: int, tau: Union[float, Tensor] = 1.0, standardize: bool = False
) -> Tensor:
    """Computes a smooth approximation to amax(X, dim=dim) with a fat tail.

    Args:
        X: A Tensor from which to compute the smoothed amax.
        tau: Temperature parameter controlling the smooth approximation
            to max operator, becomes tighter as tau goes to 0. Needs to be positive.
        standardize: Toggles the temperature standardization of the smoothed function.

    Returns:
        A Tensor of smooth approximations to max(X, dim=dim).
    """
    if x.shape[dim] == 1:
        return x.squeeze(dim)
    M = x.amax(dim=dim, keepdim=True)
    if standardize:  # (M - m) could also scale by range
        # force scaling < 1, could actually also let this blow up if we're primarily
        # concerned with relative precision.
        tau = tau * x.std(dim=dim, keepdim=True).clamp_max(1)  # detach?
    y = (x - M) / tau
    M = M.squeeze(dim)
    if isinstance(tau, Tensor) and tau.ndim > M.ndim:  # happens e.g. when standardizing
        tau = tau.squeeze(dim)
    return M + tau * cauchy(y).sum(dim=dim).log()


def cauchy(x: Tensor) -> Tensor:
    return 1 / (1 + x.square())


def check_dtype_float32_or_float64(X: Tensor) -> None:
    if X.dtype != torch.float32 and X.dtype != torch.float64:
        raise UnsupportedError(
            f"Only dtypes float32 and float64 are supported, but received {X.dtype}."
        )


def check_tau(tau: Union[float, Tensor]) -> Union[float, Tensor]:
    """Checks the validity of the tau arguments of the functions below, and returns tau
    if it is valid."""
    if isinstance(tau, Tensor) and tau.numel() != 1:
        raise ValueError(f"tau is not a scalar: {tau.numel() = }.")
    if not (tau > 0):
        raise ValueError(f"tau is non-positive: {tau = }.")
    return tau


################################## constraint utils #######################################
def compute_log_constraint_indicator(
    constraints: List[Callable[[Tensor], Tensor]],
    samples: Tensor,
    eta: Tensor,
    fatten: bool,
) -> Tensor:
    r"""Computes the feasibility indicator of a list of constraints given posterior
    samples, using a log-sigmoid to smoothly approximate the feasibility indicator
    of each individual constraint to ensure differentiability and high gradient signal.

    Args:
        constraints: A list of callables, each mapping a Tensor of size `b x q x m`
            to a Tensor of size `b x q`, where negative values imply feasibility.
            This callable must support broadcasting. Only relevant for multi-
            output models (`m` > 1).
        samples: A `n_samples x b x q x m` Tensor of samples drawn from the posterior.
        eta: The temperature parameter for the sigmoid function. Can be either a float
            or a 1-dim tensor. In case of a float the same eta is used for every
            constraint in constraints. In case of a tensor the length of the tensor
            must match the number of provided constraints. The i-th constraint is
            then estimated with the i-th eta value.
        fatten: Toggles the logarithmic / linear asymptotic behavior of the log sigmoid.

    Returns:
        A `n_samples x b x q`-dim tensor of log smoothed feasibility indicator values.
    """
    if type(eta) != Tensor:
        eta = torch.full((len(constraints),), eta)
    if len(eta) != len(constraints):
        raise ValueError(
            "Number of provided constraints and number of provided etas do not match."
        )
    is_feasible = torch.zeros_like(samples[..., 0])
    log_sigmoid = log_fatmoid if fatten else logexpit
    for constraint, e in zip(constraints, eta):
        # standardize input to log_sigmoid here?
        is_feasible = is_feasible + log_sigmoid(-constraint(samples) / e)

    return is_feasible


def logexpit(X: Tensor) -> Tensor:
    """Computes the logarithm of the expit (a.k.a. sigmoid) function."""
    return -log1pexp(-X)


def log1pexp(x: Tensor) -> Tensor:
    """Numerically accurate evaluation of log(1 + exp(x)).
    See [Maechler2012accurate]_ for details.
    """
    mask = x <= 18
    return torch.where(
        mask,
        (lambda z: z.exp().log1p())(x.masked_fill(~mask, 0)),
        (lambda z: z + (-z).exp())(x.masked_fill(mask, 0)),
    )


def log_fatmoid(x: Tensor, tau: Union[float, Tensor] = 1.0) -> Tensor:
    return fatmoid(x, tau=tau).log()


def fatmoid(x: Tensor, tau: Union[float, Tensor] = 1.0) -> Tensor:
    """Computes a twice continuously differentiable approximation to the Heaviside
    step function with a fat tail, i.e. O(1 / x^2) as x goes to -inf.

    Args:
        x: A Tensor from which to compute the smoothed step function.
        tau: Temperature parameter controlling the smoothness of the approximation.
        TODO: consider standardization.
        if standardize:
            tau = tau * x.std(dim=?, keepdim=True)
    """
    x = x / tau
    m = math.sqrt(1 / 3)  # this defines the inflection point
    return torch.where(
        x < 0,
        2 / 3 * cauchy(x - m),
        1 - 2 / 3 * cauchy(x + m),
    )
