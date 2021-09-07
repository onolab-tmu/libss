"""Auxiliary-function-based independent vector analysis."""
import numpy as np

from .update_rules import update_spatial_model, update_covariance
from .utils import demix, tensor_H

eps = np.finfo(np.float64).eps


class AuxIVA(object):
    """
    Base class for auxiliary-function-based independent vector analysis.

    Attributes
    ----------
    observations : ndarray of shape (n_frame, n_freq, n_src)
    update_demix_filter : str, default="IP1"
        Update method of demixing filter.
        Only "IP1" and "IP2" are available for now.
    update_source_model : str, default="Gauss"
        The source model, i.e. prior information of source signals.
        Only "Gauss" and "Laplace" are available for now.

    estimated : ndarray of shape (n_frame, n_freq, n_src)
    source_model : ndarray of shape (n_frame, n_freq, n_src)
    demix_filter : ndarray of shape (n_freq, n_src, n_src)
    loss : list[float]
    """

    def __init__(
        self,
        observations,
        update_demix_filter="IP1",
        update_source_model="Laplace",
        **kwargs,
    ):
        """Initialize parameters in AuxIVA."""
        # Setup
        self.observations = observations
        self.update_demix_filter = update_demix_filter
        self.update_source_model = update_source_model
        self.kwargs = kwargs

        # Results
        self.demix_filter = self.init_demix()
        self.estimated = demix(self.observations, self.demix_filter)
        self.source_model = None
        self.covariance = None
        self.loss = self.calc_loss()

    def step(self):
        """Update paramters one step."""
        n_src = self.observations.shape[-1]
        # 1. Update source model
        self.source_model = self.calc_source_model()

        # 2. Update covariance
        self.covariance = update_covariance(self.observations, self.source_model)

        # 2. Update demixing filter
        if self.update_demix_filter in ["IP1", "ISS1"]:
            for s in range(n_src):
                self.demix_filter[:, :, :] = update_spatial_model(
                    self.covariance,
                    self.demix_filter,
                    row_idx=s,
                    method=self.update_demix_filter,
                )
        elif self.update_demix_filter in ["IP2"]:
            for s in range(0, n_src * 2, 2):
                m, n = s % n_src, (s + 1) % n_src
                tgt_idx = np.array([m, n])
                self.demix_filter[:, :, :] = update_spatial_model(
                    self.covariance,
                    self.demix_filter,
                    row_idx=tgt_idx,
                    method=self.update_demix_filter,
                )
        else:
            raise NotImplementedError

        # 3. Update estimated sources
        self.estimated = demix(self.observations, self.demix_filter)

        # 4. Update loss function value
        self.loss = self.calc_loss()

    def init_demix(self):
        """Initialize demixing matrix."""
        _, n_freq, n_src = self.observations.shape
        W0 = np.zeros((n_freq, n_src, n_src), dtype=complex)
        W0[:, :, :n_src] = np.tile(np.eye(n_src, dtype=complex), (n_freq, 1, 1))
        return W0

    def calc_source_model(self):
        """
        Calculate source model.

        Returns
        -------
        ndarray of shape (n_frame, n_freq, n_src)
        """
        n_freq = self.observations.shape[1]
        allowed = {
            "Gauss": lambda y: (np.linalg.norm(y, axis=1) ** 2) / n_freq,
            "Laplace": lambda y: 2.0 * np.linalg.norm(y, axis=1),
        }

        # (n_frame, n_src)
        y_norm = allowed[self.update_source_model](self.estimated)

        # (n_frame, n_freq, n_src)
        source_model = np.zeros(self.observations.shape)
        source_model[:, :, :] = np.maximum(eps, y_norm)[:, None, :]

        return source_model

    def calc_loss(self):
        """Calculate loss function value."""
        n_frames, _, _ = self.estimated.shape

        f_norm = lambda y: np.linalg.norm(y, axis=1)
        contrast_func = {
            "Laplace": lambda y: np.sum(f_norm(y)),
            "Gauss": lambda y: np.sum(np.log(1.0 / np.maximum(eps, f_norm(y)))),
        }[self.update_source_model]
        target_loss = contrast_func(self.estimated)

        tfn_fnt = [1, 2, 0]
        XX = self.observations.transpose(tfn_fnt)
        YY = self.estimated.transpose(tfn_fnt)
        W_H = np.linalg.solve(XX @ tensor_H(XX), XX @ tensor_H(YY))
        _, logdet = np.linalg.slogdet(W_H)
        demix_loss = -2 * n_frames * np.sum(logdet)

        return target_loss + demix_loss
