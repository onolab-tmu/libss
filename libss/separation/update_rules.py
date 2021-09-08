"""Update paramters."""
import numpy as np
from .utils import tensor_H, solve_2x2HEAD

NP_EPS = np.finfo(np.float64).eps


def residual_2x2head(u1, u2, V1, V2):
    """
    Print the residual of HEAD problem.

    Parameters
    ----------
    u1, u2: (n_freq, 2)
    V1, V2: (n_freq, 2, 2)

    Returns
    -------
    Residual matrix of HEAD with the given parameters.
    """
    # TODO: refactor
    head11 = (u1[:, None, :].conj() @ V1 @ u1[:, :, None]).squeeze().real.mean()
    head21 = (u2[:, None, :].conj() @ V1 @ u1[:, :, None]).squeeze().real.mean()

    head12 = (u1[:, None, :].conj() @ V2 @ u2[:, :, None]).squeeze().real.mean()
    head22 = (u2[:, None, :].conj() @ V2 @ u2[:, :, None]).squeeze().real.mean()

    return np.array([[head11, head12], [head21, head22]])


def update_source_model(y_power, B, A, eps=NP_EPS):
    """
    Update source model parameters `B`, `A` corresponding one source with multiplicative update rules.

    Parameters
    ----------
    y_power: ndarray (n_freq, n_frame)
        Power spectrograms of demixed signals.
    B: ndarray (n_freq, n_basis)
        Basis matrices of source spectrograms.
    A: ndarray (n_frame, n_basis)
        Activity matrices of source spectrograms.
    eps: float, optional
        `B` and `A` are pre-processed `B[B < eps] = eps` to improve numerical stability.
        Default is `np.finfo(np.float64).eps`.

    Returns
    -------
    Updated B and A
    """
    R = B @ A.T
    iR = np.reciprocal(R)

    B *= (y_power * np.square(iR)) @ A / (iR @ A)
    B[B < eps] = eps

    R = B @ A.T
    iR = np.reciprocal(R)

    A *= (y_power.T * np.square(iR.T)) @ B / (iR.T @ B)
    A[A < eps] = eps

    return B, A


def _ip_1(cov, demix, row_idx):
    """
    Update one demixing vector with IP1 algorithm.

    Parameters
    ----------
    cov: ndarray (n_src, n_freq, n_src, n_src)
        Weighted covariance matrices of the observed signal.
    demix: ndarray (n_freq, n_src, n_src)
        Demixing filters.
    row_idx: int
        The index of row vector of W.

    Returns
    -------
    Updated W
    """
    n_src = cov.shape[-1]

    # shape: (n_freq, n_src, n_src)
    w = (np.linalg.solve(demix @ cov[row_idx], np.eye(n_src)[None, :, row_idx])).conj()
    denom = (w[:, None, :] @ cov[row_idx]) @ w[:, :, None].conj()
    demix[:, row_idx, :] = w / np.sqrt(denom[:, :, 0])

    return demix


def _ip_2(cov, demix, row_idx, verbose=False):
    """
    Update two demixing vectors with IP-2 algorithm.

    Parameters
    ----------
    cov: ndarray (n_src, n_freq, n_src, n_src)
        Weighted covariance matrices of the observed signal.
    demix: ndarray (n_freq, n_src, n_src)
        Demixing filters.
    row_idx: ndarray (2,)
        The indeces of row vector of W.

    Returns
    -------
    Updated W
    """
    n_src = cov.shape[-1]

    # shape: (2, n_freq, n_src, 2)
    proj = np.linalg.solve(
        demix[None, :, :, :] @ cov[row_idx], np.eye(n_src)[None, None, :, row_idx]
    )

    # shape: (2, n_freq, 2, 2)
    U = tensor_H(proj) @ cov[row_idx] @ proj

    # Eigen vectors of U[1] @ inv(U[0])
    # shape: (2, n_freq, 2)
    _, u = solve_2x2HEAD(U[0], U[1])
    if verbose:
        print(residual_2x2head(u[0], u[1], U[0], U[1]))

    demix[:, row_idx, :, None] = (proj @ u[:, :, :, None]).swapaxes(0, 1).conj()

    return demix


def _iss_1(cov, demix, row_idx, flooring=True, eps=NP_EPS):
    """
    Update all demixing vectors with ISS algorithm.

    Parameters
    ----------
    cov: ndarray (n_freq, n_src, n_src)
        Weighted covariance matrices of the observed signal.
    demix: ndarray (n_freq, n_src, n_src)
        Demixing filters.
    row_idx: int
        The index of row vector of W.
    flooring: bool, optional
        If True, flooring is processed.
        Default is True.
    eps : float, option (default `np.finfo(np.float64).eps`)

    Returns
    -------
    Updated W
    """
    raise NotImplementedError


def update_covariance(observed, source_model, prev_cov=None, alpha=None):
    """
    Update weighted covariance matrices with given observed signal and source model.

    Parameters
    ----------
    observed : ndarray (n_frame, n_freq, n_src)
    source_model : ndarray (n_frame, n_freq, n_src)

    Returns
    -------
    cov : ndarray (n_src, n_freq, n_src, n_src)
    """
    n_frame, n_freq, n_src = observed.shape
    observed_t = observed.transpose([1, 2, 0])

    # (n_src, n_freq, n_frame)
    model_t = source_model.transpose([2, 1, 0])

    cov = np.zeros((n_src, n_freq, n_src, n_src), dtype=complex)

    # shape: (n_src, n_freq, n_src, n_src)
    for s in range(n_src):
        cov[s] = (observed_t / model_t[s, :, None, :]) @ tensor_H(observed_t)
    cov /= n_frame

    # block batch or online
    if prev_cov is not None and alpha is not None:
        cov *= 1 - alpha
        cov += alpha * prev_cov

    return cov


def update_spatial_model(cov, demix, row_idx, method="IP1"):
    """
    Update demixing matrix W.

    Parameters
    ----------
    cov : ndarray (n_src, n_freq, n_src, n_src)
        Variance matrices of source spectrograms.
    demix : ndarray (n_freq, n_src, n_src)
        Demixing matrices.
    row_idx : int or ndarray (2,)
        The index of row vector of W.
    method : string

    Returns
    -------
    Updated W
    """
    update = {
        "IP1": _ip_1,
        "IP2": _ip_2,
        "ISS1": _iss_1,
    }[method]

    return update(cov, demix.copy(), row_idx)
