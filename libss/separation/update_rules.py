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


def _ip_1(x, R, W, row_idx):
    """
    Update one demixing vector with IP1 algorithm.

    Parameters
    ----------
    x: ndarray (n_freq, n_src, n_frame)
        Power spectrograms of demixed signals.
    R: ndarray (n_src, n_freq, n_frame)
        Variance matrices of source spectrograms.
    W: ndarray (n_freq, n_src, n_src)
        Demixing filters.
    row_idx: int
        The index of row vector of W.

    Returns
    -------
    Updated W
    """
    _, n_src, n_frame = x.shape

    # shape: (n_freq, n_src, n_src)
    cov = (x / R[row_idx, :, None, :]) @ tensor_H(x) / n_frame
    w = (np.linalg.solve(W @ cov, np.eye(n_src)[None, :, row_idx])).conj()
    denom = (w[:, None, :] @ cov) @ w[:, :, None].conj()
    W[:, row_idx, :] = w / np.sqrt(denom[:, :, 0])

    return W


def _ip_2(x, R, W, row_idx, verbose=False):
    """
    Update two demixing vectors with IP-2 algorithm.

    Parameters
    ----------
    x: ndarray (n_freq, n_src, n_frame)
        Power spectrograms of demixed signals.
    R: ndarray (n_src, n_freq, n_frame)
        Variance matrices of source spectrograms.
    W: ndarray (n_freq, n_src, n_src)
        Demixing filters.
    row_idx: ndarray (2,)
        The indeces of row vector of W.

    Returns
    -------
    Updated W
    """
    _, n_src, n_frames = x.shape

    # shape: (2, n_freq, n_src, n_src)
    V = (
        (x[None, :, :, :] / R[row_idx, :, None, :])
        @ tensor_H(x[None, :, :, :])
        / n_frames
    )

    # shape: (2, n_freq, n_src, 2)
    P = np.linalg.solve(W[None, :, :, :] @ V, np.eye(n_src)[None, None, :, row_idx])

    # shape: (2, n_freq, 2, 2)
    U = tensor_H(P) @ V @ P

    # Eigen vectors of U[1] @ inv(U[0])
    # shape: (2, n_freq, 2)
    _, u = solve_2x2HEAD(U[0], U[1])
    if verbose:
        print(residual_2x2head(u[0], u[1], U[0], U[1]))

    W[:, row_idx, :, None] = (P @ u[:, :, :, None]).swapaxes(0, 1).conj()

    return W


def _iss_1(x, R, W, row_idx, flooring=True, eps=NP_EPS):
    """
    Update all demixing vectors with ISS algorithm.

    Parameters
    ----------
    x: ndarray (n_freq, n_src, n_frame)
        Power spectrograms of demixed signals.
    R: ndarray (n_src, n_freq, n_frame)
        Variance matrices of source spectrograms.
    W: ndarray (n_freq, n_src, n_src)
        Demixing filters.
    row_idx: int
        The index of row vector of W.
    flooring: bool, optional
        If True, flooring is processed.
        Default is True.

    Returns
    -------
    Updated W
    """
    y = W @ x
    n_freq, n_src, n_frame = y.shape

    # (n_freq, n_src, n_frame)
    iR = np.reciprocal(R.transpose([1, 0, 2]))

    # update coefficients of W
    v_denom = np.zeros((n_freq, n_src), dtype=x.dtype)
    v = np.zeros((n_freq, n_src), dtype=x.dtype)

    # separation
    v[:, :, None] = (y * iR) @ y[:, row_idx, :, None].conj()
    v_denom[:, :, None] = iR @ np.square(np.abs(y[:, row_idx, :, None]))

    # flooring is processed to improve numerical stability
    if flooring:
        v_denom[v_denom < eps] = eps
    v[:, :] /= v_denom
    v[:, row_idx] -= 1 / np.sqrt(v_denom[:, row_idx] / n_frame)

    # update demixing matrices and demixed signals
    W[:, :, :] -= v[:, :, None] * W[:, row_idx, None, :]
    y[:, :, :] -= v[:, :, None] * y[:, row_idx, None, :]

    return W


def update_spatial_model(x, R, W, row_idx, method="IP1"):
    """
    Update demixing matrix W.

    Parameters
    ----------
    x: ndarray (n_frame, n_freq, n_src)
        Input mixture signal.
    R: ndarray (n_frame, n_freq, n_src)
        Variance matrices of source spectrograms.
    W: ndarray (n_freq, n_src)
        Demixing matrices.
    row_idx: int or ndarray (2,)
        The index of row vector of W.
    method: string

    Returns
    -------
    Updated W
    """
    allowed_methods = {
        "IP1": _ip_1,
        "IP2": _ip_2,
        "ISS1": _iss_1,
    }

    # Transpose matrices to calculate efficiently
    xt = x.transpose([1, 2, 0])  # (n_freq, n_src, n_frame)
    Rt = R.transpose([2, 1, 0])  # (n_src, n_freq, n_frame)

    return allowed_methods[method](xt, Rt, W.copy(), row_idx)
