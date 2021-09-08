"""Collection of utility functions for ICA-based BSS."""
import numpy as np

from .metrics import si_bss_eval


def tensor_T(A):
    """Compute transpose for tensor."""
    return A.swapaxes(-2, -1)


def tensor_H(A):
    """Compute Hermitian transpose for tensor."""
    return np.conj(A).swapaxes(-2, -1)


def cost_ilrma(source, basis, activ, demix, axis=None):
    """
    Calculate the cost function of ILRMA.

    Parameters
    ----------
    source: ndarray (n_frame, n_freq, n_src)
        分離信号
    basis: ndarray (n_src, n_freq, n_basis)
        基底
    activ: ndarray (n_src, n_frame, n_basis)
        アクティビティ
    demix: ndarray (n_freq, n_mic, n_mic)
        音源分離・残響除去フィルタ
    axis: None, or tuple of int

    Returns
    -------
    The value of cost function of ILRMA

    Raises
    ------
    ValueError:
        If `cost` is infinite or not a number.
    """
    # (n_freq, n_src, n_frame)
    s_fnt = source.transpose([1, 2, 0])

    # (n_freq, n_src, n_frame)
    src_pwr = np.square(abs(s_fnt))

    # (n_src, n_freq, n_frame) -> (n_freq, n_src, n_frame)
    src_var = (basis @ tensor_T(activ)).transpose([1, 0, 2])

    # (n_freq,)
    spt_term = -2 * np.linalg.slogdet(demix)[1]

    # (n_freq, n_frame)
    src_term = np.sum(src_pwr / src_var + np.log(src_var), axis=1)

    cost = np.sum(src_term + spt_term[:, None], axis=axis)
    if np.isinf(cost).any() or np.isnan(cost).any():
        raise ValueError("Cost cannot be calculated.")
    else:
        return cost


def normalize(power_spectrograms, demixing_matrix, basis_matrix, source_model):
    """
    Normalize ILRMA paramters.

    Parameters
    ----------
    power_spectrograms: ndarray (n_freq, n_src, n_frame)
        Power spectrogram of demixed signals.
    demixing_matrix: ndarray (n_freq, n_src, n_src*(n_tap+1))
        Demixing filters.
    basis_matrix: ndarray (n_src, n_freq, n_basis)
        Basis matrices of source spectrograms.
    source_model: ndarray (n_src, n_freq, n_frame)
        Variance of source model.

    Returns
    -------
    Normalized paramters.
    """
    # shape: (n_src,)
    coef = power_spectrograms.mean(axis=(0, 2))

    demixing_matrix /= np.sqrt(coef[None, :, None])
    power_spectrograms /= coef[None, :, None]
    basis_matrix /= coef[:, None, None]
    source_model /= coef[:, None, None]


def projection_back(Y, X, ref_mic=0, W=None):
    """
    Solves the scale ambiguity according to Murata et al., 2001.

    This technique uses the steering vector value corresponding
    to the demixing matrix obtained during separation.

    Parameters
    ----------
    Y: array_like (n_frames, n_freq, n_chan)
        The STFT data to project back on the reference signal
    X: array_like (n_frames, n_freq, n_chan*(n_tap+1))
        The reference signal
    ref_mic: int, optional
        The index of reference microphone.
        Default is 0.
    W: array_like (n_freq, n_chan, n_chan*(n_tap+1))
        The demixing/dereverberating matrices.
        If given, the projection matrices is culculated more efficiently.
        Default is None.
    Returns
    -------
    Y: array_like (n_frames, n_freq, n_chan)
        The projected data
    """
    n_src = Y.shape[-1]

    if W is not None:
        A = W.copy()
        invW = np.linalg.inv(W[:, :n_src, :n_src])
        A[:, :n_src, :n_src] = invW
        A[:, :n_src, n_src:] = -invW @ W[:, :n_src, n_src:]
    else:
        # find a bunch of non-zero frames
        I_nz = np.linalg.norm(Y, axis=(1, 2)) > 0

        # non-zero frames of X and Y
        # (n_freq, n_tap_src, n_frame)
        X_nz = X[I_nz, :, :].transpose([1, 2, 0]).copy()
        Y_nz = X_nz.copy()
        Y_nz[:, :n_src, :] = Y[I_nz, :, :].transpose([1, 2, 0])

        # (n_freq, n_tap_src, n_tap_src)
        A = X_nz @ tensor_H(Y_nz) @ np.linalg.inv(Y_nz @ tensor_H(Y_nz))

    # (n_freq, n_chan, n_frame)
    return Y * A[:, ref_mic, :n_src, None].transpose([2, 0, 1])


def head_error(W, R, X):
    """
    Calculate a distance between constraints of the HEAD problem and demixing matrices.

    HEAD問題の拘束条件と、各種更新手法で得られた分離行列との差を計算

    Parameters
    ----------
    W: ndarray (n_freq, n_src, n_src)
        The demixing matrices.
    R: ndarray (n_src, n_freq, n_frame)
        The weighted covariance matrices.
    X: ndarray (n_freq, n_src, n_frame)
        Input mixture.

    Returns
    -------
    Residual error of `np.eye(n_src) - WU` by using Flobenius norm
    """
    n_freq, n_src, n_frame = X.shape

    U = np.zeros((n_src, n_freq, n_src, n_src), dtype=X.dtype)
    UW = np.zeros((n_freq, n_src, n_src), dtype=U.dtype)
    for s in range(n_src):
        U[s] = (X / R[s, :, None, :]) @ tensor_H(X) / n_frame
        UW[:, :, s, None] = U[s] @ np.conj(W[:, s, :, None])

    # shape: (n_freq, src, src)
    head_matrix = W @ UW

    # calculate residual error (Flobenius norm)
    return np.linalg.norm(np.eye(n_src)[None, :, :] - head_matrix, axis=(1, 2)).mean()


def optimal_demix(S, X):
    """
    Calculate an optimal demixing matrix with the original source images and the observed mixture.

    Parameters
    ----------
    S: ndarray (n_frame, n_freq, n_src)
        音源信号 source image
    X: ndarray (n_frame, n_freq, n_mic)
        観測信号

    Returns
    -------
    W_opt: ndarray (n_freq, n_src, n_mic)
    """
    S_f = S.transpose([1, 2, 0])  # (n_freq, n_src, n_frame)
    X_f = X.transpose([1, 2, 0])  # (n_freq, n_src, n_frame)

    # Extract non-zero frames of S
    # to avoid singular matrix error
    non_zero = np.linalg.norm(S, axis=(1, 2)) > 0
    X_nz = X_f[:, :, non_zero]
    S_nz = S_f[:, :, non_zero]

    return S_nz @ tensor_H(X_nz) @ np.linalg.inv(X_nz @ tensor_H(X_nz))


def phase_invariant_norm(X, Y, axis=None):
    """
    Calculate Frobenius-norm between two complex matrices without phase rotation.

    位相回転を無視した，2個の複素行列間のFrobeniusノルムを計算.

    Parameters
    ----------
    X: ndarray (n_freq, n_mic, n_src)
    Y: ndarray (n_freq, n_mic, n_src)
    axis: {None, int, or 2-tuple of ints}, optional
        If axis is an integer, it specifies the axis of x along which to compute the vector norms.
        If axis is a 2-tuple, it specifies the axes that hold 2-D matrices, and the matrix norms of these matrices are computed.
        If axis is None then either a vector norm (when x is 1-D) or a matrix norm (when x is 2-D) is returned.
        The default is None.

    Returns
    -------
    norm: float
        Frobenius norm of $X - diag(d) Y$
    d: ndarray (n_freq, n_src, n_src)
        Rotation coefficient
    """
    n_freq, n_src, _ = X.shape
    d = np.zeros((n_freq, n_src), dtype=complex)

    for n in range(n_src):
        z = tensor_H(X[:, n, :, None]) @ Y[:, n, :, None]
        d[:, n, None, None] = np.exp(-1j * np.angle(z))

    d_mat = np.array([np.diag(d[f, :]) for f in range(n_freq)])

    return np.linalg.norm(X - d_mat @ Y, axis=axis), d_mat


def solve_2x2HEAD(V1, V2, method="ono", eig_reverse=False):
    """
    Solve a 2x2 HEAD problem with given two positive semi-definite matrices.

    Parameters
    ----------
    V1: (n_freq, 2, 2)
    V2: (n_freq, 2, 2)
    method: "numpy" or "ono"
        If "numpy", `eigval` is calculated by using `numpy.linalg.eig`.
        If "ono", `eigval` is calculated by the method presented in Ono2012IWAENC.
    eig_reverse: bool
        If True, eigenvalues is sorted in *ascending* order.
        Default is False.
        This parameter will be deprecated in the future.

    Returns
    -------
    eigval: (n_freq, 2)
        eigenvalues, must be real numbers
    eigvec: (2, n_freq, 2)
        eigenvectors corresponding to the eigenvalues
    """
    V_hat = np.array([V1, V2])

    if method == "numpy":
        # shape: (n_freq, 2, 2)
        Z = np.linalg.solve(V1, V2)

        # eigval.shape: (n_freq, 2)
        # eigvec.shape: (2, n_freq, 2)
        eigval, eigvec = np.linalg.eig(Z)

        # sorting in descending order by default
        if not eig_reverse:
            I_inv = eigval[:, 1] < eigval[:, 0]
        else:
            I_inv = eigval[:, 1] > eigval[:, 0]
        eigval[I_inv, :] = eigval[I_inv, ::-1]
        eigvec[I_inv, :, :] = eigvec[I_inv, :, ::-1]
        eigvec = eigvec.transpose([2, 0, 1])

        # (2, n_freq, 2)
        eigvec[:, :, :, None] /= np.sqrt(
            np.conj(eigvec[:, :, None, :]) @ V_hat @ eigvec[:, :, :, None]
        )
    elif method == "ono":
        # shape: (n_freq, 2, 2)
        Z = np.zeros(V1.shape, dtype=complex)

        # Z = adj(V1) @ V2
        Z[:, 0, 0] = V2[:, 0, 0] * V1[:, 1, 1] - V2[:, 1, 0] * V1[:, 0, 1]
        Z[:, 0, 1] = V2[:, 0, 1] * V1[:, 1, 1] - V2[:, 1, 1] * V1[:, 0, 1]
        Z[:, 1, 0] = -V2[:, 0, 0] * V1[:, 1, 0] + V2[:, 1, 0] * V1[:, 0, 0]
        Z[:, 1, 1] = -V2[:, 0, 1] * V1[:, 1, 0] + V2[:, 1, 1] * V1[:, 0, 0]

        # shape: (n_freq,)
        tr_Z = np.trace(Z, axis1=1, axis2=2)
        dd = np.sqrt(tr_Z ** 2 - 4 * np.linalg.det(Z))

        # shape: (n_freq,)
        d1 = (tr_Z + dd).real
        d2 = (tr_Z - dd).real

        # shape: (n_freq, 2)
        eigval = np.array([d1, d2]).T

        D = np.zeros(Z.shape, dtype=Z.dtype)
        D[:, [0, 1], [0, 1]] = np.array([d1, d2]).T

        # shape: (2, n_freq, 2)
        eigvec = (2 * Z - D).transpose([2, 0, 1])
        eigvec[:, :, :, None] /= np.sqrt(
            np.conj(eigvec[:, :, None, :]) @ V_hat @ eigvec[:, :, :, None]
        )
    else:
        raise ValueError

    return eigval, eigvec


def demix(observations, demix_filter):
    """
    Perform the demixing filter into observations.

    Parameters
    ----------
    observations : ndarray of shape (n_frame, n_freq, n_src)
    demix_filter : ndarray of shape (n_freq, n_src, n_src)

    Returns
    -------
    Estimated source
        ndarray of shape (n_frame, n_freq, n_src)
    """
    # shape: (n_freq, n_src, n_frame)
    y = demix_filter @ observations.transpose([1, 2, 0])
    return y.transpose([2, 0, 1])


def callback_eval(ref, tar, SDR, SIR, SAR):
    """
    Measure SI-SDR, SI-SIR, and SI-SAR.

    Parameters
    ----------
    ref: ndarray(n_chan, n_sample)
    tar: ndarray(n_chan, n_sample)
    SDR: list
    SIR: list
    SAR: list

    Returns
    -------
    Permutated signal.
    """
    si_sdr, si_sir, si_sar, si_perm = si_bss_eval(ref[:, :].T, tar[:, :].T)

    SDR.append(si_sdr.tolist())
    SIR.append(si_sir.tolist())
    SAR.append(si_sar.tolist())

    return tar[si_perm, :]


def whiten(x):
    """
    Whiten input signal.

    Parameters
    ----------
    x : ndarray of shape (n_frame, n_freq, n_src)
        Input multi-channel spectrograms

    Returns
    -------
    Whitened input.
    """
    x0 = x - x.mean(axis=1)[:, None, :]
    # (n_freq, n_frame, n_src)
    x_ = x0.transpose([1, 0, 2])

    # (n_freq, n_src, n_src)
    V = np.mean(x_[:, :, :, None] @ x_[:, :, None, :].conj(), axis=1)

    val, mat = np.linalg.eigh(V)
    D = np.array([np.diag(val[s]) for s in range(val.shape[0])])
    y_ = np.linalg.inv(np.sqrt(D)) @ mat.swapaxes(-1, -2).conj() @ x_.swapaxes(-1, -2)

    return y_.transpose([2, 0, 1])
