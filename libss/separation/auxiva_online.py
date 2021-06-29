"""Auxiliary-function-based independent vector analysis."""
import numpy as np

from .update_rules import update_spatial_model, update_covariance
from .utils import demix

eps = np.finfo(np.float64).eps


def auxiva_online(
    mix,
    update_demix_filter="IP1",
    update_source_model="Laplace",
    block_size=1,
    forget_param=0.97,
    n_iter=2,
):
    """Separate with online AuxIVA."""
    # initialize
    n_frame, n_freq, n_src = mix.shape
    W = np.zeros((n_frame, n_freq, n_src, n_src), dtype=complex)
    W[:, :, :, :] = np.tile(np.eye(n_src), (n_frame, n_freq, 1, 1))

    estimated = demix(mix, W[0])
    r = np.zeros((n_frame, n_src), dtype=complex)
    cov = np.zeros((n_frame, n_src, n_freq, n_src, n_src), dtype=complex)

    np.random.seed(2)
    cov += 1e-9 * np.random.rand(*cov.shape)

    for t in range(0, n_frame):
        if t != 0:
            W[t] = W[t - 1].copy()

        for _ in range(n_iter):
            for s in range(n_src):
                r[t, s, None, None] = np.linalg.norm(
                    W[t, :, s, None, :] @ mix[t, :, :, None], axis=0
                )

                cov[t, s, :, :, :] = (
                    mix[t, :, :, None] @ mix[t, :, None, :].conj()
                ) / r[t, s]
                cov[t, s, :, :, :] = (
                    forget_param * cov[t - 1, s, :, :, :]
                    + (1 - forget_param) * cov[t, s, :, :]
                )
                cov[t, s, :, :, :] = np.maximum(cov[t, s, :, :, :], np.finfo(float).eps)

                w_ast = np.linalg.solve(
                    W[t, :, :, :] @ cov[t, s, :, :, :],
                    np.eye(n_src)[None, :, s],
                )
                W[t, :, s, :] = w_ast.conj()
                denom = np.sqrt(
                    w_ast[:, None, :] @ cov[t, s, :, :, :] @ w_ast[:, :, None].conj()
                ).squeeze()
                W[t, :, s, :] /= denom[:, None]

    for t in range(n_frame):
        estimated[t, :, :] = demix(mix[t, None, :, :], W[t])

    return W, estimated


class OnlineAuxIVA(object):
    """
    Base class for auxiliary-function-based independent vector analysis.

    Attributes
    ----------
    observed : ndarray of shape (n_frame, n_freq, n_src)
    update_demix_filter : str
    update_source_model : str, {"Gauss", "Laplace"}
    block_size : int, default 1
    forget_param : float, default 0.97
        Forgetting parameter for autoregressive calculation of covariance matrices.
        Real value over `0 < forget_param <= 1`
    update_covariance : str

    estimated : ndarray of shape (n_frame, n_freq, n_src)
    source_model : ndarray of shape (n_frame, n_freq, n_src)
    demix_filter : ndarray of shape (n_frame+1, n_freq, n_src, n_src)
    covariance : ndarray of shape (n_frame+1, n_src, n_freq, n_src, n_src)
    loss : list[float]
    """

    def __init__(
        self,
        observed,
        update_demix_filter="IP1",
        update_source_model="Laplace",
        block_size=1,
        forget_param=0.97,
        n_iter=2,
        **kwargs,
    ):
        """Initialize parameters in AuxIVA."""
        # Setup
        self.observed = observed
        self.update_demix_filter = update_demix_filter
        self.update_source_model = update_source_model
        self.block_size = block_size
        self.forget_param = forget_param
        self.n_iter = n_iter
        self.kwargs = kwargs

        # Results
        n_frame, n_freq, n_src = observed.shape
        self.demix_filter = self.init_demix()
        self.estimated = demix(self.observed, self.demix_filter[0, :, :, :])
        self.covariance = np.zeros(
            (n_frame - block_size + 1, n_src, n_freq, n_src, n_src), dtype=complex
        )
        self.source_model = None
        self.loss = None

    def step(self):
        """Update paramters one step."""
        lb = self.block_size
        n_frame, _, n_src = self.observed.shape
        for i, t in enumerate(range(lb, n_frame)):
            for _ in range(self.n_iter):
                # 1. Update source model
                self.source_model = self.calc_source_model(
                    self.estimated[t - lb : t + 1, :, :], self.update_source_model
                )

                for s in range(n_src):
                    # TODO: Accelerate with matrix inversion lemma
                    # 2. Update covariance matrices
                    if t == lb:  # at the beginning of iteration
                        self.covariance[i] = update_covariance(
                            self.observed[t - lb : t + 1, :, :], self.source_model
                        )
                    else:
                        self.covariance[i] = update_covariance(
                            self.observed[t - lb : t + 1, :, :],
                            self.source_model,
                            prev_cov=self.covariance[i - lb],
                            alpha=self.forget_param,
                        )

                    # TODO: Fix IP2
                    # 3. Update demixing filter
                    self.demix_filter[i] = update_spatial_model(
                        self.covariance[i],
                        self.demix_filter[i],
                        row_idx=s,
                        method=self.update_demix_filter,
                    )

                # 4. Update estimated sources
                self.estimated[t - lb : t + 1, :, :] = demix(
                    self.observed[t - lb : t + 1, :, :],
                    self.demix_filter[i],
                )

    def init_demix(self):
        """Initialize demixing matrix."""
        n_frame, n_freq, n_src = self.observed.shape
        W0 = np.zeros((n_frame, n_freq, n_src, n_src), dtype=complex)
        W0[:, :, :n_src] = np.tile(
            np.eye(n_src, dtype=complex), (n_frame, n_freq, 1, 1)
        )
        return W0

    def calc_source_model(self, estimated, model):
        """
        Calculate source model.

        Parameters
        ----------
        demix_filter : ndarray (n_freq, n_src, n_src)
        estimated : ndarray (n_frame, n_freq, n_src)

        Returns
        -------
        ndarray of shape (n_frame, n_freq, n_src)
        """
        n_freq = estimated.shape[1]
        f_norm = {
            "Gauss": lambda y: (np.linalg.norm(y, axis=1) ** 2) / n_freq,
            "Laplace": lambda y: 2.0 * np.linalg.norm(y, axis=1),
        }[model]

        # (n_frame, n_src)
        y_norm = f_norm(estimated)

        # (n_block, n_freq, n_src)
        source_model = np.zeros(estimated.shape)
        source_model[:, :, :] = np.maximum(eps, y_norm)[:, None, :]

        return source_model
