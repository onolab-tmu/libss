"""Auxiliary-function-based independent vector analysis."""
import numpy as np

from .update_rules import update_spatial_model, update_covariance
from .utils import demix

eps = np.finfo(np.float64).eps


def projection_back_frame(W, ref_mic=0):
    """
    Perform projection-back technique for online AuxIVA algorithm.

    Parameters
    ----------
    W : ndarray (n_frame, n_freq, n_src, n_src)
    ref_mic : int, default=0
        Index of reference microphone
    """
    n_freq, _, _ = W.shape
    W_proj = W.copy()

    A = np.linalg.inv(W)
    for f in range(n_freq):
        eA = np.diagflat(A[f, ref_mic, :])
        W_proj[f, :, :] = eA @ W[f, :, :]
    return W_proj


def auxiva_online(
    mix,
    update_demix_filter="IP1",
    update_source_model="Laplace",
    block_size=1,
    forget_param=0.97,
    n_iter=2,
    ref_mic=0,
):
    """
    Separate with online AuxIVA.

    Parameters
    ----------
    mix : ndarray (n_frame, n_freq, n_src)
    update_demix_filter : str, {'IP1'}
        Update method of demixing matrices.
        Note: only 'IP1' is available currently.
    update_source_model : str, {'Laplace', 'Gauss'}
    block_size : int, default=1
    forget_param : float, default=0.97
    n_iter : int, optional default=2
    ref_mic : int, default=0

    Returns
    -------
    The demixing matrices and separated sources.
    """
    # initialize
    n_frame, n_freq, n_src = mix.shape
    eye = np.eye(n_src, dtype=complex)
    W = np.tile(eye, (n_frame, n_freq, 1, 1))
    cov = np.tile(1e-5 * eye, (n_frame, n_src, n_freq, 1, 1))

    cont = {
        "Gauss": lambda y: np.linalg.norm(y, axis=0) / n_freq,
        "Laplace": lambda y: 2.0 * np.linalg.norm(y, axis=0),
    }[update_source_model]

    # Iteration
    for t in range(0, n_frame):
        W[t] = W[t - 1].copy()

        for _ in range(n_iter):
            for s in range(n_src):
                # Update source model
                est = W[t, :, s, None, :] @ mix[t, :, :, None]
                r = cont(est)

                # Update weighted covariance
                cov_new = mix[t, :, :, None] @ mix[t, :, None, :].conj() / r
                cov[t, s] = forget_param * cov[t - 1, s] + (1 - forget_param) * cov_new

                # Update demixing vector
                w_ = np.linalg.solve(W[t] @ cov[t, s], eye[None, :, s])
                denom = np.sqrt(w_[:, None, :] @ cov[t, s] @ w_[:, :, None].conj())
                W[t, :, s, :] = w_.conj() / denom[:, 0, 0, None]

        W[t] = projection_back_frame(W[t])

    # Calculate output signal
    estimated = np.zeros(mix.shape, dtype=complex)
    for t in range(n_frame):
        estimated[t] = demix(mix[t, None, :, :], W[t])

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
        self.demix_filter = np.tile(
            np.eye(n_src, dtype=complex), (n_frame, n_freq, 1, 1)
        )
        self.estimated = demix(self.observed, self.demix_filter[0, :, :, :])
        self.covariance = np.tile(
            np.eye(n_src, dtype=complex), (n_frame, n_src, n_freq, 1, 1)
        )
        self.source_model = np.zeros(self.estimated.shape)
        for t in range(self.block_size):
            self.source_model[t, :, :] = self.calc_source_model(
                self.estimated[t, None, :, :], self.update_source_model
            )[:, None, :]

        self.loss = None

    def step(self):
        """Update paramters one step."""
        lb = self.block_size
        n_frame, _, n_src = self.observed.shape
        for t in range(lb, n_frame):
            self.demix_filter[t] = self.demix_filter[t - 1].copy()
            for _ in range(self.n_iter):
                # 1. Update source model
                self.source_model[t, :, :] = self.calc_source_model(
                    self.estimated[t, None, :, :], self.update_source_model
                )[:, None, :]

                for s in range(n_src):
                    # TODO: Accelerate with matrix inversion lemma
                    # 2. Update covariance matrices
                    self.covariance[t] = update_covariance(
                        self.observed[t - lb : t],
                        self.source_model[t - lb : t],
                        prev_cov=self.covariance[t - lb],
                        alpha=self.forget_param,
                    )

                    # TODO: Fix IP2
                    # 3. Update demixing filter
                    self.demix_filter[t] = update_spatial_model(
                        self.covariance[t],
                        self.demix_filter[t],
                        row_idx=s,
                        method=self.update_demix_filter,
                    )

                # 4. Update estimated sources
                self.estimated[t, None, :, :] = demix(
                    self.observed[t, None, :, :],
                    self.demix_filter[t],
                )

    def calc_source_model(self, estimated, model):
        """Calculate source model."""
        n_freq = estimated.shape[1]
        f_norm = {
            "Gauss": lambda y: (np.linalg.norm(y, axis=1) ** 2) / n_freq,
            "Laplace": lambda y: 2.0 * np.linalg.norm(y, axis=1),
        }[model]

        # (n_frame, n_src)
        y_norm = f_norm(estimated)

        return y_norm
