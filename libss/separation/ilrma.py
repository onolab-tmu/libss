"""Independent low-rank matrix analysis."""
import numpy as np

from .update_rules import update_spatial_model, update_covariance
from .utils import demix, tensor_T

eps = np.finfo(np.float64).eps


class ILRMA(object):
    """
    Base class for independent low-rank matrix analysis.

    Attributes
    ----------
    observations : ndarray of shape (n_frame, n_freq, n_src)
    update_demix_filter : str, default="IP1"
        Update method of demixing filter.
        Only "IP1" and "IP2" are available for now.
    update_source_model : str, default="Gauss"
        The source model, i.e. prior information of source signals.
        Only "Gauss" is available for now.

    estimated : ndarray of shape (n_frame, n_freq, n_src)
    source_model : ndarray of shape (n_frame, n_freq, n_src)
    demix_filter : ndarray of shape (n_freq, n_src, n_src)
    loss : list[float]
    """

    def __init__(
        self,
        observations,
        update_demix_filter="IP1",
        update_source_model="Gauss",
        **kwargs,
    ):
        """Initialize parameters in ILRMA."""
        # Setup
        self.observations = observations
        self.update_demix_filter = update_demix_filter
        self.update_source_model = update_source_model
        self.params = kwargs

        # Results
        self.demix_filter = self.init_demix()
        self.estimated = demix(self.observations, self.demix_filter)
        self.basis = self.init_basis()
        self.activ = self.init_activ()
        self.source_model = self.init_source_model()
        self.covariance = None
        self.loss = self.calc_loss()

    def step(self):
        """Update paramters one step."""
        n_src = self.observations.shape[-1]
        y_power = np.square(np.abs(self.estimated))

        # 2. Update demixing filter
        if self.update_demix_filter in ["IP1", "ISS1"]:
            # 1. Update source model
            for s in range(n_src):
                b = self.basis[s]
                a = self.activ[s]
                self.basis[s], self.activ[s] = self.calc_source_model(
                    b, a, y_power[:, :, s]
                )
                self.source_model[:, :, s] = self.activ[s] @ self.basis[s].T

            # 2. Update covariance
            self.covariance = update_covariance(self.observations, self.source_model)

            # 3. Update demixing matrix
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
                for l in tgt_idx:
                    b = self.basis[l]
                    a = self.activ[l]
                    self.basis[l], self.activ[l] = self.calc_source_model(
                        b, a, y_power[:, :, l]
                    )
                    self.source_model[:, :, l] = self.activ[l] @ self.basis[l].T

                # 2. Update covariance
                self.covariance = update_covariance(
                    self.observations, self.source_model
                )

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

    def init_basis(self):
        """Initialize basis matrix."""
        n_basis = self.params["n_basis"]
        _, n_freq, n_src = self.observations.shape
        return np.ones((n_src, n_freq, n_basis))

    def init_activ(self):
        """Initialize activation matrix."""
        n_basis = self.params["n_basis"]
        n_frame, _, n_src = self.observations.shape
        return np.random.uniform(low=0.1, high=1.0, size=(n_src, n_frame, n_basis))

    def init_source_model(self):
        """Initialize demixing matrix."""
        # TODO: fix order of dimensions to remove `np.transpose`
        # (n_src, n_freq, n_frame) -> (n_frame, n_freq, n_src)
        r = self.basis @ tensor_T(self.activ)
        return r.transpose([2, 1, 0])

    def calc_source_model(self, B, A, y_power):
        """
        Calculate source model.
        By overriding this method, various source models (e.g., Student t, ILRMA-T, generalized Kullback---Leibler divergence, or IDLMA) can be applied.

        Parameters
        ----------
        B : ndarray of shape (n_freq, n_basis)
            Basis matrix
        A : ndarray of shape (n_frame, n_basis)
            Activation matrix
        y_power : ndarray of shape (n_frame, n_freq)
            Power spectrograms of estimated source

        Returns
        -------
        """
        # TODO: fix this ad-hoc process
        y_power = y_power.T

        R = B @ A.T
        iR = np.reciprocal(R)

        B *= (y_power * np.square(iR)) @ A / (iR @ A)
        B[B < eps] = eps

        R = B @ A.T
        iR = np.reciprocal(R)

        A *= (y_power.T * np.square(iR.T)) @ B / (iR.T @ B)
        A[A < eps] = eps

        return B, A

    def calc_loss(self, axis=None):
        """
        Calculate loss function value of ILRMA.

        Parameters
        ----------
        axis : int or None, default=None

        Raises
        ------
        ValueError:
            If `cost` is infinite or not a number.
        """
        # (n_frame, n_freq, n_src)
        y_power = np.square(np.abs(self.estimated))

        # basis: (n_src, n_freq, n_basis)
        # activ: (n_src, n_frame, n_basis)

        # (n_src, n_freq, n_frame) -> (n_frame, n_freq, n_src)
        src_var = (self.basis @ tensor_T(self.activ)).transpose([2, 1, 0])

        # (n_freq,)
        target_loss = -2 * np.linalg.slogdet(self.demix_filter)[1]

        # (n_frame, n_freq)
        demix_loss = np.sum(y_power / src_var + np.log(src_var), axis=2)

        cost = np.sum(demix_loss + target_loss[None, :], axis=axis)
        if np.isinf(cost).any() or np.isnan(cost).any():
            raise ValueError("Cost cannot be calculated.")
        else:
            return cost
