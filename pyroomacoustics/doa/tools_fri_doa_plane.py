from __future__ import division, print_function
import datetime
import time
import numpy as np
import scipy as sp
from scipy import linalg
import scipy.special
import scipy.optimize
from functools import partial
import os


def polar2cart(rho, phi):
    """
    convert from polar to cartesian coordinates

    Parameters
    ----------
    rho: 
        radius
    phi: 
        azimuth
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y



def cov_mtx_est(y_mic):
    """
    estimate covariance matrix

    Parameters
    ----------
    y_mic: 
        received signal (complex based band representation) at microphones
    """
    # Q: total number of microphones
    # num_snapshot: number of snapshots used to estimate the covariance matrix
    '''
    Q, num_snapshot = y_mic.shape
    cov_mtx = np.zeros((Q, Q), dtype=complex, order='F')
    for q in range(Q):
        y_mic_outer = y_mic[q, :]
        for qp in range(Q):
            y_mic_inner = y_mic[qp, :]
            cov_mtx[qp, q] = np.dot(y_mic_outer, y_mic_inner.T.conj())
    '''
    cov_mtx = np.dot(np.conj(y_mic), y_mic.T)
    return cov_mtx / y_mic.shape[1]


def extract_off_diag(mtx):
    """
    extract off diagonal entries in mtx.
    The output vector is order in a column major manner.

    Parameters
    ----------
    mtx: 
        input matrix to extract the off diagonal entries
    """
    # we transpose the matrix because the function np.extract will first flatten the matrix
    # withe ordering convention 'C' instead of 'F'!!
    extract_cond = np.reshape((1 - np.eye(*mtx.shape)).T.astype(bool), (-1, 1), order='F')
    return np.reshape(np.extract(extract_cond, mtx.T), (-1, 1), order='F')


def multiband_cov_mtx_est(y_mic):
    """
    estimate covariance matrix based on the received signals at microphones

    Parameters
    ----------
    y_mic: 
        received signal (complex base-band representation) at microphones
    """
    # Q: total number of microphones
    # num_snapshot: number of snapshots used to estimate the covariance matrix
    # num_bands: number of sub-bands considered
    Q, num_snapshot, num_bands = y_mic.shape
    cov_mtx = np.zeros((Q, Q, num_bands), dtype=complex)
    for band in range(num_bands):
        for q in range(Q):
            y_mic_outer = y_mic[q, :, band]
            for qp in range(Q):
                y_mic_inner = y_mic[qp, :, band]
                cov_mtx[qp, q, band] = np.dot(y_mic_outer, y_mic_inner.T.conj())
    return cov_mtx / num_snapshot


def multiband_extract_off_diag(mtx):
    """
    extract off-diagonal entries in mtx
    The output vector is order in a column major manner

    Parameters
    ----------
    mtx: input matrix to extract the off-diagonal entries
    """
    # we transpose the matrix because the function np.extract will first flatten the matrix
    # withe ordering convention 'C' instead of 'F'!!
    Q = mtx.shape[0]
    num_bands = mtx.shape[2]
    extract_cond = np.reshape((1 - np.eye(Q)).T.astype(bool), (-1, 1), order='F')
    return np.column_stack([np.reshape(np.extract(extract_cond, mtx[:, :, band].T),
                                       (-1, 1), order='F')
                            for band in range(num_bands)])


def mtx_freq2raw(M, p_mic_x, p_mic_y):
    """
    build the matrix that maps the Fourier series to the raw microphone signals

    Parameters
    ----------
    M: 
        the Fourier series expansion is limited from -M to M
    p_mic_x: 
        a vector that contains microphones x coordinates
    p_mic_y: 
        a vector that contains microphones y coordinates
    """
    num_mic = p_mic_x.size
    ms = np.reshape(np.arange(-M, M + 1, step=1), (1, -1), order='F')
    G = np.zeros((num_mic, 2 * M + 1), dtype=complex, order='C')
    count_G = 0
    for q in range(num_mic):
        norm_q = np.sqrt(p_mic_x[q] ** 2 + p_mic_y[q] ** 2)
        phi_q = np.arctan2(p_mic_y[q], p_mic_x[q])
        G[q, :] = (-1j) ** ms * sp.special.jv(ms, norm_q) * \
                  np.exp(1j * ms * phi_q)
    return G


def mtx_freq2visi(M, p_mic_x, p_mic_y):
    """
    build the matrix that maps the Fourier series to the visibility

    Parameters
    ----------
    M: 
        the Fourier series expansion is limited from -M to M
    p_mic_x: 
        a vector that constains microphones x coordinates
    p_mic_y: 
        a vector that constains microphones y coordinates
    """
    num_mic = p_mic_x.size
    ms = np.reshape(np.arange(-M, M + 1, step=1), (1, -1), order='F')
    p_mic_x_outer = p_mic_x[:, np.newaxis]
    p_mic_y_outer = p_mic_y[:, np.newaxis]
    p_mic_x_inner = p_mic_x[np.newaxis, :]
    p_mic_y_inner = p_mic_y[np.newaxis, :]
    extract_cond = np.reshape((1 - np.eye(num_mic)).astype(bool), (-1, 1), order='C')
    baseline_x = np.extract(extract_cond, p_mic_x_outer - p_mic_x_inner)[:, np.newaxis]
    baseline_y = np.extract(extract_cond, p_mic_y_outer - p_mic_y_inner)[:, np.newaxis]
    baseline_norm = np.sqrt(baseline_x * baseline_x + baseline_y * baseline_y)
    return (-1j) ** ms * sp.special.jv(ms, baseline_norm) * \
           np.exp(1j * ms * np.arctan2(baseline_y, baseline_x))


def mtx_fri2signal_ri_multiband(M, p_mic_x_all, p_mic_y_all, D1, D2, aslist=False, signal='visibility'):
    """
    build the matrix that maps the Fourier series to the visibility in terms of
    REAL-VALUED entries only. (matrix size double)

    Parameters
    ----------
    M: 
        the Fourier series expansion is limited from -M to M
    p_mic_x_all: 
        a matrix that contains microphones x coordinates
    p_mic_y_all: 
        a matrix that contains microphones y coordinates
    D1: 
        expansion matrix for the real-part
    D2: 
        expansion matrix for the imaginary-part aslist: whether the linear
        mapping for each subband is returned as a list or a block diagonal
        matrix
    signal: 
        The type of signal considered ('visibility' for covariance matrix, 'raw' for microphone inputs)
    """
    num_bands = p_mic_x_all.shape[1]
    if aslist:
        return [mtx_fri2signal_ri(M, p_mic_x_all[:, band_count],
                                  p_mic_y_all[:, band_count], D1, D2, signal)
                for band_count in range(num_bands)]
    else:
        return linalg.block_diag(*[mtx_fri2signal_ri(M, p_mic_x_all[:, band_count],
                                                     p_mic_y_all[:, band_count], D1, D2, signal=signal)
                                   for band_count in range(num_bands)])


def mtx_fri2signal_ri(M, p_mic_x, p_mic_y, D1, D2, signal='visibility'):
    """
    build the matrix that maps the Fourier series to the visibility in terms of
    REAL-VALUED entries only. (matrix size double)

    Parameters
    ----------
    M: 
        the Fourier series expansion is limited from -M to M
    p_mic_x: 
        a vector that contains microphones x coordinates
    p_mic_y: 
        a vector that contains microphones y coordinates
    D1: 
        expansion matrix for the real-part
    D2: 
        expansion matrix for the imaginary-part
    signal: 
        The type of signal considered ('visibility' for covariance matrix, 'raw' for microphone inputs)
    """

    if signal == 'visibility':
        func = mtx_freq2visi
    elif signal == 'raw':
        func = mtx_freq2raw

    return np.dot(cpx_mtx2real(func(M, p_mic_x, p_mic_y)),
                  linalg.block_diag(D1, D2))


def cpx_mtx2real(mtx):
    """
    extend complex valued matrix to an extended matrix of real values only

    Parameters
    ----------
    mtx: 
        input complex valued matrix
    """
    return np.vstack((np.hstack((mtx.real, -mtx.imag)), np.hstack((mtx.imag, mtx.real))))


def hermitian_expan(half_vec_len):
    """
    expand a real-valued vector to a Hermitian symmetric vector.
    The input vector is a concatenation of the real parts with NON-POSITIVE indices and
    the imaginary parts with STRICTLY-NEGATIVE indices.

    Parameters
    ----------
    half_vec_len: 
        length of the first half vector
    """
    D0 = np.eye(half_vec_len)
    D1 = np.vstack((D0, D0[1:, ::-1]))
    D2 = np.vstack((D0, -D0[1:, ::-1]))
    D2 = D2[:, :-1]
    return D1, D2


def output_shrink(K, L):
    """
    shrink the convolution output to half the size.
    used when both the annihilating filter and the uniform samples of sinusoids satisfy
    Hermitian symmetric.

    Parameters
    ----------
    K: 
        the annihilating filter size: K + 1
    L: 
        length of the (complex-valued) b vector
    """
    out_len = L - K
    if out_len % 2 == 0:
        half_out_len = np.int(out_len / 2.)
        mtx_r = np.hstack((np.eye(half_out_len),
                           np.zeros((half_out_len, half_out_len))))
        mtx_i = mtx_r
    else:
        half_out_len = np.int((out_len + 1) / 2.)
        mtx_r = np.hstack((np.eye(half_out_len),
                           np.zeros((half_out_len, half_out_len - 1))))
        mtx_i = np.hstack((np.eye(half_out_len - 1),
                           np.zeros((half_out_len - 1, half_out_len))))
    return linalg.block_diag(mtx_r, mtx_i)


def coef_expan_mtx(K):
    """
    expansion matrix for an annihilating filter of size K + 1

    Parameters
    ----------
    K: 
        number of Dirac. The filter size is K + 1
    """
    if K % 2 == 0:
        D0 = np.eye(np.int(K / 2. + 1))
        D1 = np.vstack((D0, D0[1:, ::-1]))
        D2 = np.vstack((D0, -D0[1:, ::-1]))[:, :-1]
    else:
        D0 = np.eye(np.int((K + 1) / 2.))
        D1 = np.vstack((D0, D0[:, ::-1]))
        D2 = np.vstack((D0, -D0[:, ::-1]))
    return D1, D2


def Tmtx_ri(b_ri, K, D, L):
    """
    build convolution matrix associated with b_ri

    Parameters
    ----------
    b_ri: 
        a real-valued vector
    K: 
        number of Diracs
    D1: 
        expansion matrix for the real-part
    D2: 
        expansion matrix for the imaginary-part
    """
    b_ri = np.dot(D, b_ri)
    b_r = b_ri[:L]
    b_i = b_ri[L:]
    Tb_r = linalg.toeplitz(b_r[K:], b_r[K::-1])
    Tb_i = linalg.toeplitz(b_i[K:], b_i[K::-1])
    return np.vstack((np.hstack((Tb_r, -Tb_i)), np.hstack((Tb_i, Tb_r))))


def Tmtx_ri_half(b_ri, K, D, L, D_coef):
    ''' Split T matrix in conjugate symmetric representation '''
    return np.dot(Tmtx_ri(b_ri, K, D, L), D_coef)


def Tmtx_ri_half_out_half(b_ri, K, D, L, D_coef, mtx_shrink):
    """
    if both b and annihilation filter coefficients are Hermitian symmetric,
    then the output will also be Hermitian symmetric => the effectively output
    is half the size
    """
    return np.dot(np.dot(mtx_shrink, Tmtx_ri(b_ri, K, D, L)), D_coef)


def Rmtx_ri(coef_ri, K, D, L):
    ''' Split T matrix in rea/imaginary representation '''
    coef_ri = np.squeeze(coef_ri)
    coef_r = coef_ri[:K + 1]
    coef_i = coef_ri[K + 1:]
    R_r = linalg.toeplitz(np.concatenate((np.array([coef_r[-1]]),
                                          np.zeros(L - K - 1))),
                          np.concatenate((coef_r[::-1],
                                          np.zeros(L - K - 1)))
                          )
    R_i = linalg.toeplitz(np.concatenate((np.array([coef_i[-1]]),
                                          np.zeros(L - K - 1))),
                          np.concatenate((coef_i[::-1],
                                          np.zeros(L - K - 1)))
                          )
    return np.dot(np.vstack((np.hstack((R_r, -R_i)), np.hstack((R_i, R_r)))), D)


def Rmtx_ri_half(coef_half, K, D, L, D_coef):
    ''' Split T matrix in rea/imaginary conjugate symmetric representation '''
    return Rmtx_ri(np.dot(D_coef, coef_half), K, D, L)


def Rmtx_ri_half_out_half(coef_half, K, D, L, D_coef, mtx_shrink):
    """
    if both b and annihilation filter coefficients are Hermitian symmetric,
    then the output will also be Hermitian symmetric => the effectively output
    is half the size
    """
    return np.dot(mtx_shrink, Rmtx_ri(np.dot(D_coef, coef_half), K, D, L))


def build_mtx_amp(phi_k, p_mic_x, p_mic_y):
    """
    the matrix that maps Diracs' amplitudes to the visibility

    Parameters
    ----------
    phi_k: 
        Diracs' location (azimuth)
    p_mic_x: 
        a vector that contains microphones' x-coordinates
    p_mic_y: 
        a vector that contains microphones' y-coordinates
    """
    xk, yk = polar2cart(1, phi_k[np.newaxis, :])
    num_mic = p_mic_x.size
    p_mic_x_outer = p_mic_x[:, np.newaxis]
    p_mic_y_outer = p_mic_y[:, np.newaxis]
    p_mic_x_inner = p_mic_x[np.newaxis, :]
    p_mic_y_inner = p_mic_y[np.newaxis, :]
    extract_cond = np.reshape((1 - np.eye(num_mic)).astype(bool), (-1, 1), order='C')
    baseline_x = np.extract(extract_cond, p_mic_x_outer - p_mic_x_inner)[:, np.newaxis]
    baseline_y = np.extract(extract_cond, p_mic_y_outer - p_mic_y_inner)[:, np.newaxis]
    return np.exp(-1j * (xk * baseline_x + yk * baseline_y))


def build_mtx_amp_ri(p_mic_x, p_mic_y, phi_k):
    ''' builds real/imaginary amplitude matrix '''
    mtx = build_mtx_amp(phi_k, p_mic_x, p_mic_y)
    return np.vstack((mtx.real, mtx.imag))


def build_mtx_raw_amp(p_mic_x, p_mic_y, phi_k):
    """
    the matrix that maps Diracs' amplitudes to the visibility

    Parameters
    ----------
    phi_k: 
        Diracs' location (azimuth)
    p_mic_x: 
        a vector that contains microphones' x-coordinates
    p_mic_y: 
        a vector that contains microphones' y-coordinates
    """
    xk, yk = polar2cart(1, phi_k)
    num_mic = p_mic_x.size
    K = phi_k.size
    mtx = np.zeros((num_mic, K), dtype=complex, order='C')
    for q in range(num_mic):
        mtx[q, :] = np.exp(-1j * (xk * p_mic_x[q] + yk * p_mic_y[q]))

    return mtx


def mtx_updated_G_multiband(phi_recon, M, mtx_amp2visi_ri,
                            mtx_fri2visi_ri, num_bands):
    """
    Update the linear transformation matrix that links the FRI sequence to the
    visibilities by using the reconstructed Dirac locations.

    Parameters
    ----------
    phi_recon: 
        the reconstructed Dirac locations (azimuths)
    M: 
        the Fourier series expansion is between -M to M
    p_mic_x: 
        a vector that contains microphones' x-coordinates
    p_mic_y: 
        a vector that contains microphones' y-coordinates
    mtx_fri2visi: 
        the linear mapping from Fourier series to visibilities
    """
    L = 2 * M + 1
    ms_half = np.reshape(np.arange(-M, 1, step=1), (-1, 1), order='F')
    phi_recon = np.reshape(phi_recon, (1, -1), order='F')
    mtx_amp2freq = np.exp(-1j * ms_half * phi_recon)  # size: (M + 1) x K
    mtx_amp2freq_ri = np.vstack((mtx_amp2freq.real, mtx_amp2freq.imag[:-1, :]))  # size: (2M + 1) x K
    mtx_fri2amp_ri = linalg.lstsq(mtx_amp2freq_ri, np.eye(L), rcond=None)[0]
    # projection mtx_freq2visi to the null space of mtx_fri2amp
    mtx_null_proj = np.eye(L) - np.dot(mtx_fri2amp_ri.T,
                                       linalg.lstsq(mtx_fri2amp_ri.T, np.eye(L))[0])
    G_updated = np.dot(mtx_amp2visi_ri,
                       linalg.block_diag(*([mtx_fri2amp_ri] * num_bands))
                       ) + \
                np.dot(mtx_fri2visi_ri,
                       linalg.block_diag(*([mtx_null_proj] * num_bands))
                       )
    return G_updated


def mtx_updated_G_multiband_new(phi_opt, M, p_x, p_y,
                                G0_lst, num_bands):
    """
    Update the linear transformation matrix that links the FRI sequence to the
    visibilities by using the reconstructed Dirac locations.

    Parameters
    ----------
    phi_opt: 
        the reconstructed Dirac locations (azimuths)
    M: 
        the Fourier series expansion is between -M to M
    p_mic_x: 
        a vector that contains microphones' x-coordinates
    p_mic_y: 
        a vector that contains microphones' y-coordinates
    G0_lst: 
        the original linear mapping from Fourier series to visibilities
    num_bands: 
        number of subbands
    """
    L = 2 * M + 1
    ms_half = np.reshape(np.arange(-M, 1, step=1), (-1, 1), order='F')

    mtx_amp2freq = np.exp(-1j * ms_half * np.reshape(phi_opt, (1, -1), order='F'))
    mtx_amp2freq_ri = np.vstack((mtx_amp2freq.real, mtx_amp2freq.imag[:-1, :]))
    mtx_fri2amp_ri = linalg.lstsq(mtx_amp2freq_ri, np.eye(L))[0]

    G_updated = []
    for band_count in range(num_bands):
        G0_loop = G0_lst[band_count]
        amp_mtx_ri_loop = \
            build_mtx_amp_ri(p_x[:, band_count],
                             p_y[:, band_count],
                             phi_opt)

        high_freq_mapping = np.dot(amp_mtx_ri_loop, mtx_fri2amp_ri)

        # G_updated.append(
        #     G0_loop +
        #     high_freq_mapping -
        #     np.dot(
        #         G0_loop,
        #         linalg.solve(np.dot(G0_loop.T, G0_loop),
        #                      np.dot(G0_loop.T, high_freq_mapping))
        #     )
        # )
        G_updated.append(
            high_freq_mapping +
            G0_loop -
            np.dot(np.dot(G0_loop, mtx_fri2amp_ri.T),
                   linalg.solve(np.dot(mtx_fri2amp_ri, mtx_fri2amp_ri.T),
                                mtx_fri2amp_ri)
                   )
        )

    return G_updated


def mtx_updated_G(phi_recon, M, mtx_amp2visi_ri, mtx_fri2visi_ri):
    """
    Update the linear transformation matrix that links the FRI sequence to the
    visibilities by using the reconstructed Dirac locations.
    Parameters
    ----------
    phi_recon: 
        the reconstructed Dirac locations (azimuths)
    M: 
        the Fourier series expansion is between -M to M
    p_mic_x: 
        a vector that contains microphones' x-coordinates
    p_mic_y: 
        a vector that contains microphones' y-coordinates
    mtx_freq2visi: 
        the linear mapping from Fourier series to visibilities
    """
    L = 2 * M + 1
    ms_half = np.reshape(np.arange(-M, 1, step=1), (-1, 1), order='F')
    phi_recon = np.reshape(phi_recon, (1, -1), order='F')
    mtx_amp2freq = np.exp(-1j * ms_half * phi_recon)  # size: (M + 1) x K
    mtx_amp2freq_ri = np.vstack((mtx_amp2freq.real, mtx_amp2freq.imag[:-1, :]))  # size: (2M + 1) x K
    mtx_fri2amp_ri = linalg.lstsq(mtx_amp2freq_ri, np.eye(L))[0]
    # projection mtx_freq2visi to the null space of mtx_fri2amp
    mtx_null_proj = np.eye(L) - np.dot(mtx_fri2amp_ri.T,
                                       linalg.lstsq(mtx_fri2amp_ri.T, np.eye(L))[0])
    G_updated = np.dot(mtx_amp2visi_ri, mtx_fri2amp_ri) + \
                np.dot(mtx_fri2visi_ri, mtx_null_proj)
    return G_updated


def dirac_recon_ri(G, a_ri, K, M, noise_level, max_ini=100, stop_cri='mse'):
    """
    Reconstruct point sources' locations (azimuth) from the visibility measurements

    Parameters
    ----------
    G: 
        the linear transformation matrix that links the visibilities to
        uniformly sampled sinusoids
    a_ri: 
        the visibility measurements
    K: 
        number of Diracs
    M: 
        the Fourier series expansion is between -M and M
    noise_level: 
        level of noise (ell_2 norm) in the measurements
    max_ini: 
        maximum number of initialisations
    stop_cri: 
        stopping criterion, either 'mse' or 'max_iter'
    """
    L = 2 * M + 1  # length of the (complex-valued) b vector
    a_ri = a_ri.flatten('F')
    compute_mse = (stop_cri == 'mse')
    # size of G: (Q(Q-1)) x (2M + 1), where Q is the number of antennas
    assert not np.iscomplexobj(G)  # G should be real-valued
    GtG = np.dot(G.T, G)
    Gt_a = np.dot(G.T, a_ri)
    # maximum number of iterations with each initialisation
    max_iter = 50
    min_error = float('inf')
    # the least-square solution
    beta_ri = linalg.lstsq(G, a_ri)[0]
    D1, D2 = hermitian_expan(M + 1)
    D = linalg.block_diag(D1, D2)
    # size of Tbeta_ri: 2(L - K) x 2(K + 1)
    Tbeta_ri = Tmtx_ri(beta_ri, K, D, L)

    # size of various matrices / vectors
    sz_G1 = L

    sz_Tb0 = 2 * (L - K)
    sz_Tb1 = 2 * (K + 1)

    sz_Rc0 = 2 * (L - K)
    sz_Rc1 = L

    sz_coef = 2 * (K + 1)
    sz_bri = L

    rhs = np.append(np.zeros(sz_coef + sz_Tb0 + sz_G1, dtype=float), 1)
    rhs_bl = np.concatenate((Gt_a, np.zeros(sz_Rc0, dtype=Gt_a.dtype)))

    # the main iteration with different random initialisations
    for ini in range(max_ini):
        c_ri = np.random.randn(sz_coef, 1)
        c0_ri = c_ri.copy()
        error_seq = np.zeros(max_iter, dtype=float)
        R_loop = Rmtx_ri(c_ri, K, D, L)
        for inner in range(max_iter):
            mtx_loop = np.vstack((np.hstack((np.zeros((sz_coef, sz_coef)),
                                             Tbeta_ri.T,
                                             np.zeros((sz_coef, sz_Rc1)),
                                             c0_ri)),
                                  np.hstack((Tbeta_ri,
                                             np.zeros((sz_Tb0, sz_Tb0)),
                                             -R_loop,
                                             np.zeros((sz_Rc0, 1))
                                             )),
                                  np.hstack((np.zeros((sz_Rc1, sz_Tb1)),
                                             -R_loop.T,
                                             GtG,
                                             np.zeros((sz_Rc1, 1))
                                             )),
                                  np.hstack((c0_ri.T,
                                             np.zeros((1, sz_Tb0 + sz_Rc1 + 1))
                                             ))
                                  ))
            # matrix should be symmetric
            mtx_loop = (mtx_loop + mtx_loop.T) / 2.
            c_ri = linalg.lstsq(mtx_loop, rhs)[0][:sz_coef]
            # c_ri = linalg.solve(mtx_loop, rhs)[:sz_coef]

            R_loop = Rmtx_ri(c_ri, K, D, L)
            mtx_brecon = np.vstack((np.hstack((GtG, R_loop.T)),
                                    np.hstack((R_loop, np.zeros((sz_Rc0, sz_Rc0))))
                                    ))
            mtx_brecon = (mtx_brecon + mtx_brecon.T) / 2.
            b_recon_ri = linalg.lstsq(mtx_brecon, rhs_bl)[0][:sz_bri]
            # b_recon_ri = linalg.solve(mtx_brecon, rhs_bl)[:sz_bri]

            error_seq[inner] = linalg.norm(a_ri - np.dot(G, b_recon_ri))
            if error_seq[inner] < min_error:
                min_error = error_seq[inner]
                b_opt = np.dot(D1, b_recon_ri[:M + 1]) + 1j * np.dot(D2, b_recon_ri[M + 1:])
                c_opt = c_ri[:K + 1] + 1j * c_ri[K + 1:]  # real and imaginary parts

            if min_error < noise_level and compute_mse:
                break

        if min_error < noise_level and compute_mse:
            break
    return c_opt, min_error, b_opt, ini


def dirac_recon_ri_half(G, a_ri, K, M, noise_level, max_ini=100, stop_cri='mse'):
    """
    Reconstruct point sources' locations (azimuth) from the visibility measurements.
    Here we enforce hermitian symmetry in the annihilating filter coefficients so that
    roots on the unit circle are encouraged.

    Parameters
    ----------
    param G: 
        the linear transformation matrix that links the visibilities to
        uniformly sampled sinusoids
    a_ri: 
        the visibility measurements
    K: 
        number of Diracs
    M: 
        the Fourier series expansion is between -M and M
    noise_level: 
        level of noise (ell_2 norm) in the measurements
    max_ini: 
        maximum number of initialisations
    stop_cri: 
        stopping criterion, either 'mse' or 'max_iter'
    """
    L = 2 * M + 1  # length of the (complex-valued) b vector
    a_ri = a_ri.flatten('F')
    compute_mse = (stop_cri == 'mse')
    # size of G: (Q(Q-1)) x (2M + 1), where Q is the number of antennas
    assert not np.iscomplexobj(G)  # G should be real-valued
    GtG = np.dot(G.T, G)
    Gt_a = np.dot(G.T, a_ri)
    # maximum number of iterations with each initialisation
    max_iter = 50
    min_error = float('inf')
    # the least-square solution
    beta_ri = linalg.lstsq(G, a_ri)[0]
    D1, D2 = hermitian_expan(M + 1)
    D = linalg.block_diag(D1, D2)
    D_coef1, D_coef2 = coef_expan_mtx(K)
    D_coef = linalg.block_diag(D_coef1, D_coef2)
    # size of Tbeta_ri: 2(L - K) x 2(K + 1)
    Tbeta_ri = Tmtx_ri_half(beta_ri, K, D, L, D_coef)

    # size of various matrices / vectors
    sz_G1 = L

    sz_Tb0 = 2 * (L - K)
    sz_Tb1 = K + 1

    sz_Rc0 = 2 * (L - K)
    sz_Rc1 = L

    sz_coef = K + 1
    sz_bri = L

    rhs = np.append(np.zeros(sz_coef + sz_Tb0 + sz_G1, dtype=float), 1)
    rhs_bl = np.concatenate((Gt_a, np.zeros(sz_Rc0, dtype=Gt_a.dtype)))

    # the main iteration with different random initialisations
    for ini in range(max_ini):
        c_ri_half = np.random.randn(sz_coef, 1)
        c0_ri_half = c_ri_half.copy()
        error_seq = np.zeros(max_iter, dtype=float)
        R_loop = Rmtx_ri_half(c_ri_half, K, D, L, D_coef)
        for inner in range(max_iter):
            mtx_loop = np.vstack((np.hstack((np.zeros((sz_coef, sz_coef)),
                                             Tbeta_ri.T,
                                             np.zeros((sz_coef, sz_Rc1)),
                                             c0_ri_half)),
                                  np.hstack((Tbeta_ri,
                                             np.zeros((sz_Tb0, sz_Tb0)),
                                             -R_loop,
                                             np.zeros((sz_Rc0, 1))
                                             )),
                                  np.hstack((np.zeros((sz_Rc1, sz_Tb1)),
                                             -R_loop.T,
                                             GtG,
                                             np.zeros((sz_Rc1, 1))
                                             )),
                                  np.hstack((c0_ri_half.T,
                                             np.zeros((1, sz_Tb0 + sz_Rc1 + 1))
                                             ))
                                  ))
            # matrix should be symmetric
            mtx_loop = (mtx_loop + mtx_loop.T) / 2.
            c_ri_half = linalg.lstsq(mtx_loop, rhs)[0][:sz_coef]
            # c_ri = linalg.solve(mtx_loop, rhs)[:sz_coef]

            R_loop = Rmtx_ri_half(c_ri_half, K, D, L, D_coef)
            mtx_brecon = np.vstack((np.hstack((GtG, R_loop.T)),
                                    np.hstack((R_loop, np.zeros((sz_Rc0, sz_Rc0))))
                                    ))
            mtx_brecon = (mtx_brecon + mtx_brecon.T) / 2.
            b_recon_ri = linalg.lstsq(mtx_brecon, rhs_bl)[0][:sz_bri]
            # b_recon_ri = linalg.solve(mtx_brecon, rhs_bl)[:sz_bri]

            error_seq[inner] = linalg.norm(a_ri - np.dot(G, b_recon_ri))
            if error_seq[inner] < min_error:
                min_error = error_seq[inner]
                b_opt = np.dot(D1, b_recon_ri[:M + 1]) + 1j * np.dot(D2, b_recon_ri[M + 1:])
                c_ri = np.dot(D_coef, c_ri_half)
                c_opt = c_ri[:K + 1] + 1j * c_ri[K + 1:]  # real and imaginary parts

            if min_error < noise_level and compute_mse:
                break

        if min_error < noise_level and compute_mse:
            break
    return c_opt, min_error, b_opt, ini

def dirac_recon_ri_half_multiband_lu(G_lst, GtG_lst, GtG_inv_lst, a_ri, K, M, max_ini=100, max_iter=50):
    """
    Here we use LU decomposition to precompute a few entries.  Reconstruct
    point sources' locations (azimuth) from the visibility measurements.  Here
    we enforce hermitian symmetry in the annihilating filter coefficients so
    that roots on the unit circle are encouraged.

    Parameters
    ----------
    G_lst: 
        a list of the linear transformation matrices that links the
        visibilities to uniformly sampled sinusoids
    a_ri: 
        the visibility measurements
    K: 
        number of Diracs
    M: 
        the Fourier series expansion is between -M and M
    noise_level: 
        level of noise (ell_2 norm) in the measurements
    max_ini: 
        maximum number of initialisations
    stop_cri: 
        stopping criterion, either 'mse' or 'max_iter'
    """
    num_bands = a_ri.shape[1]  # number of bands considered
    L = 2 * M + 1  # length of the (complex-valued) b vector for each band
    assert not np.iscomplexobj(np.concatenate(G_lst))  # G should be real-valued
    assert not np.iscomplexobj(np.concatenate(a_ri))  # a_ri should be real-valued

    D1, D2 = hermitian_expan(M + 1)
    D = linalg.block_diag(D1, D2)
    D_coef1, D_coef2 = coef_expan_mtx(K)
    D_coef = linalg.block_diag(D_coef1, D_coef2)

    # shrink the output size to half as both the annihilating filter coeffiicnets and
    # the uniform samples of sinusoids are Hermitian symmetric
    mtx_shrink = output_shrink(K, L)

    # compute Tbeta and GtG for each subband
    Gt_a_lst = []
    beta_ri_lst = []
    Tbeta_ri_lst = []
    for loop in range(num_bands):
        #G_loop = G_lst[loop]
        #a_loop = a_ri[:, loop]

        #lu_GtG_loop = linalg.lu_factor(np.dot(G_loop.T, G_loop), check_finite=False)

        Gt_a_loop = np.dot(G_lst[loop].T, a_ri[:, loop])

        #beta_ri_loop = linalg.lu_solve(lu_GtG_loop, Gt_a_loop, check_finite=False)
        beta_ri_loop = np.dot(GtG_inv_lst[loop], Gt_a_loop)

        beta_ri_lst.append(beta_ri_loop)
        Gt_a_lst.append(Gt_a_loop)

        Tbeta_ri_lst.append(Tmtx_ri_half_out_half(
            beta_ri_loop, K, D, L, D_coef, mtx_shrink))

    sz_coef = K + 1
    rhs = np.append(np.zeros(sz_coef, dtype=float), 1)
    min_error = float('inf')

    for ini in range(max_ini):
        c_ri_half = np.random.randn(sz_coef, 1)
        c0_ri_half = c_ri_half.copy()
        Rmtx_band = Rmtx_ri_half_out_half(c_ri_half, K, D, L, D_coef, mtx_shrink)

        mtx_loop = \
            np.vstack((
                np.hstack((
                    lu_compute_mtx_obj_initial(GtG_inv_lst, Tbeta_ri_lst,
                                               Rmtx_band, num_bands, K),
                    c0_ri_half)),
                np.append(c0_ri_half, 0).T
            ))
        for inner in range(max_iter):
            # update mtx_loop
            if inner != 0:
                # mtx_loop[:sz_coef, :sz_coef] = \
                #     lu_compute_mtx_obj(Tbeta_ri_lst, num_bands, K, lu_R_GtGinv_Rt_loop)
                mtx_loop[:sz_coef, :sz_coef] = mtx_loop_upper_left
            try:
                c_ri_half = linalg.solve(mtx_loop, rhs, check_finite=False)[:sz_coef]
            except linalg.LinAlgError:
                break

            # build R based on the updated c
            Rmtx_band = Rmtx_ri_half_out_half(c_ri_half, K, D, L, D_coef, mtx_shrink)

            # update b_recon
            error_loop, mtx_loop_upper_left = \
                compute_obj_val(GtG_inv_lst, Tbeta_ri_lst, Rmtx_band, c_ri_half, num_bands, K)

            if error_loop < min_error:
                min_error = error_loop
                # b_opt = np.dot(D1, b_recon_ri[:M + 1, :]) + \
                #         1j * np.dot(D2, b_recon_ri[M + 1:, :])
                Rmtx_opt = Rmtx_band
                c_ri = np.dot(D_coef, c_ri_half)
                c_opt = c_ri[:K + 1] + 1j * c_ri[K + 1:]  # real and imaginary parts

    b_opt_ri = compute_b(G_lst, GtG_lst, beta_ri_lst, Rmtx_opt, num_bands, a_ri, use_lu=True, GtG_inv_lst=GtG_inv_lst)[0]
    b_opt = np.dot(D1, b_opt_ri[:M + 1, :]) + 1j * np.dot(D2, b_opt_ri[M + 1:, :])

    return c_opt, min_error, b_opt


def dirac_recon_ri_half_multiband(G_lst, a_ri, K, M, max_ini=100):
    """
    Reconstruct point sources' locations (azimuth) from the visibility measurements.
    Here we enforce hermitian symmetry in the annihilating filter coefficients so that
    roots on the unit circle are encouraged.

    Parameters
    ----------
    G_lst: 
        a list of the linear transformation matrices that links the
        visibilities to uniformly sampled sinusoids
    a_ri: 
        the visibility measurements
    K: 
        number of Diracs
    M: 
        the Fourier series expansion is between -M and M
    noise_level: 
        level of noise (ell_2 norm) in the measurements
    max_ini: 
        maximum number of initialisations
    stop_cri: 
        stopping criterion, either 'mse' or 'max_iter'
    """
    num_bands = a_ri.shape[1]  # number of bands considered
    L = 2 * M + 1  # length of the (complex-valued) b vector for each band
    assert not np.iscomplexobj(np.concatenate(G_lst))  # G should be real-valued
    assert not np.iscomplexobj(np.concatenate(a_ri))  # a_ri should be real-valued

    # maximum number of iterations with each initialisation
    max_iter = 50

    D1, D2 = hermitian_expan(M + 1)
    D = linalg.block_diag(D1, D2)
    D_coef1, D_coef2 = coef_expan_mtx(K)
    D_coef = linalg.block_diag(D_coef1, D_coef2)

    # shrink the output size to half as both the annihilating filter coeffiicnets and
    # the uniform samples of sinusoids are Hermitian symmetric
    mtx_shrink = output_shrink(K, L)

    # compute Tbeta and GtG for each subband
    Gt_a_lst = []
    GtG_lst = []
    beta_ri_lst = []
    Tbeta_ri_lst = []
    for loop in range(num_bands):
        G_loop = G_lst[loop]
        a_loop = a_ri[:, loop]
        beta_ri_loop = linalg.lstsq(G_loop, a_loop)[0]

        beta_ri_lst.append(beta_ri_loop)
        Gt_a_lst.append(np.dot(G_loop.T, a_loop))
        GtG_lst.append(np.dot(G_loop.T, G_loop))

        Tbeta_ri_lst.append(Tmtx_ri_half_out_half(
            beta_ri_loop, K, D, L, D_coef, mtx_shrink))

    sz_coef = K + 1
    rhs = np.append(np.zeros(sz_coef, dtype=float), 1)
    min_error = float('inf')

    for ini in range(max_ini):
        c_ri_half = np.random.randn(sz_coef, 1)
        c0_ri_half = c_ri_half.copy()
        Rmtx_band = Rmtx_ri_half_out_half(c_ri_half, K, D, L, D_coef, mtx_shrink)
        mtx_loop = np.vstack((np.hstack((compute_mtx_obj(GtG_lst, Tbeta_ri_lst,
                                                         Rmtx_band, num_bands, K),
                                         c0_ri_half)),
                              np.append(c0_ri_half, 0).T
                              ))
        for inner in range(max_iter):
            # update mtx_loop
            if inner != 0:
                mtx_loop[:sz_coef, :sz_coef] = \
                    compute_mtx_obj(GtG_lst, Tbeta_ri_lst, Rmtx_band, num_bands, K)

            # matrix should be symmetric
            mtx_loop += mtx_loop.T
            mtx_loop *= 0.5
            c_ri_half = linalg.solve(mtx_loop, rhs, check_finite=False)[:sz_coef]

            # build R based on the updated c
            Rmtx_band = Rmtx_ri_half_out_half(c_ri_half, K, D, L, D_coef, mtx_shrink)

            # update b_recon
            b_recon_ri, error_loop = \
                compute_b(G_lst, GtG_lst, beta_ri_lst, Rmtx_band, num_bands, a_ri)

            if error_loop < min_error:
                min_error = error_loop
                b_opt = np.dot(D1, b_recon_ri[:M + 1, :]) + \
                        1j * np.dot(D2, b_recon_ri[M + 1:, :])
                c_ri = np.dot(D_coef, c_ri_half)
                c_opt = c_ri[:K + 1] + 1j * c_ri[K + 1:]  # real and imaginary parts
    return c_opt, min_error, b_opt


def lu_compute_mtx_obj(Tbeta_lst, num_bands, K, lu_R_GtGinv_Rt_lst):
    """
    compute the matrix (M) in the objective function:
        min   c^H M c
        s.t.  c0^H c = 1

    Parameters
    ----------
    GtG_lst: 
        list of G^H * G
    Tbeta_lst: 
        list of Teoplitz matrices for beta-s
    Rc0: 
        right dual matrix for the annihilating filter (same of each block -> not a list)
    """
    mtx = np.zeros((K + 1, K + 1), dtype=float)  # <= assume G, Tbeta and Rc0 are real-valued

    for loop in range(num_bands):
        Tbeta_loop = Tbeta_lst[loop]
        mtx += np.dot(Tbeta_loop.T,
                      linalg.lu_solve(lu_R_GtGinv_Rt_lst[loop],
                                      Tbeta_loop, check_finite=False)
                      )

    return mtx


def lu_compute_mtx_obj_initial(GtG_inv_lst, Tbeta_lst, Rc0, num_bands, K):
    """
    compute the matrix (M) in the objective function:
        min   c^H M c
        s.t.  c0^H c = 1

    Parameters
    ----------
    GtG_lst: 
        list of G^H * G
    Tbeta_lst: 
        list of Teoplitz matrices for beta-s
    Rc0: 
        right dual matrix for the annihilating filter (same of each block -> not a list)
    """
    mtx = np.zeros((K + 1, K + 1), dtype=float)  # <= assume G, Tbeta and Rc0 are real-valued
    for loop in range(num_bands):
        Tbeta_loop = Tbeta_lst[loop]
        mtx += np.dot(Tbeta_loop.T,
                      linalg.solve(np.dot(Rc0,
                                          np.dot(GtG_inv_lst[loop], Rc0.T)),
                                   Tbeta_loop)
                      )
    return mtx


def compute_mtx_obj(GtG_lst, Tbeta_lst, Rc0, num_bands, K):
    """
    compute the matrix (M) in the objective function:
        min   c^H M c
        s.t.  c0^H c = 1

    Parameters
    ----------
    GtG_lst: 
        list of G^H * G
    Tbeta_lst: 
        list of Teoplitz matrices for beta-s
    Rc0: 
        right dual matrix for the annihilating filter (same of each block -> not a list)
    """
    mtx = np.zeros((K + 1, K + 1), dtype=float)  # <= assume G, Tbeta and Rc0 are real-valued
    for loop in range(num_bands):
        Tbeta_loop = Tbeta_lst[loop]
        GtG_loop = GtG_lst[loop]
        mtx += np.dot(Tbeta_loop.T,
                      linalg.solve(np.dot(Rc0, linalg.solve(GtG_loop, Rc0.T)),
                                   Tbeta_loop)
                      )
    return mtx

def compute_obj_val(GtG_inv_lst, Tbeta_lst, Rc0, c_ri_half, num_bands, K):
    """
    compute the fitting error.
    CAUTION: Here we assume use_lu = True
    """
    mtx = np.zeros((K + 1, K + 1), dtype=float)  # <= assume G, Tbeta and Rc0 are real-valued
    fitting_error = 0
    for band_count in range(num_bands):
        #GtG_loop = GtG_lst[band_count]
        Tbeta_loop = Tbeta_lst[band_count]

        mtx += np.dot(Tbeta_loop.T,
                      linalg.solve(
                          np.dot(Rc0, 
                              #linalg.lu_solve(GtG_loop, Rc0.T, check_finite=False)
                              np.dot(GtG_inv_lst[band_count], Rc0.T)
                              ),
                          Tbeta_loop, check_finite=False
                      )
                      )

    fitting_error += np.sqrt(np.dot(c_ri_half.T, np.dot(mtx, c_ri_half)).real)

    return fitting_error, mtx


def compute_b(G_lst, GtG_lst, beta_lst, Rc0, num_bands, a_ri, use_lu=False, GtG_inv_lst=None):
    """
    compute the uniform sinusoidal samples b from the updated annihilating
    filter coeffiients.

    Parameters
    ----------
    GtG_lst: 
        list of G^H G for different subbands
    beta_lst: 
        list of beta-s for different subbands
    Rc0: 
        right-dual matrix, here it is the convolution matrix associated with c
    num_bands: 
        number of bands
    a_ri: 
        a 2D numpy array. each column corresponds to the measurements within a subband
    """
    b_lst = []
    a_Gb_lst = []
    if use_lu:
        assert GtG_inv_lst is not None
        lu_lst = []
        for loop in range(num_bands):
            GtG_loop = GtG_lst[loop]
            beta_loop = beta_lst[loop]
            lu_loop = linalg.lu_factor(
                np.dot(Rc0, np.dot(GtG_inv_lst[loop], Rc0.T)),
                check_finite=False
            )
            b_loop = beta_loop - \
                     np.dot(GtG_inv_lst[loop],
                                     np.dot(Rc0.T,
                                            linalg.lu_solve(lu_loop, np.dot(Rc0, beta_loop),
                                                            check_finite=False)),
                                     )

            b_lst.append(b_loop)
            a_Gb_lst.append(a_ri[:, loop] - np.dot(G_lst[loop], b_loop))
            lu_lst.append(lu_loop)

        return np.column_stack(b_lst), linalg.norm(np.concatenate(a_Gb_lst)), lu_lst
    else:
        for loop in range(num_bands):
            GtG_loop = GtG_lst[loop]
            beta_loop = beta_lst[loop]
            b_loop = beta_loop - \
                     linalg.solve(GtG_loop,
                                  np.dot(Rc0.T,
                                         linalg.solve(np.dot(Rc0, linalg.solve(GtG_loop, Rc0.T)),
                                                      np.dot(Rc0, beta_loop)))
                                  )

            b_lst.append(b_loop)
            a_Gb_lst.append(a_ri[:, loop] - np.dot(G_lst[loop], b_loop))

        return np.column_stack(b_lst), linalg.norm(np.concatenate(a_Gb_lst))


def dirac_recon_ri_half_multiband_parallel(G, a_ri, K, M, max_ini=100):
    """
    Reconstruct point sources' locations (azimuth) from the visibility measurements.
    Here we enforce hermitian symmetry in the annihilating filter coefficients so that
    roots on the unit circle are encouraged.
    We use parallel implementation when stop_cri == 'max_iter'

    Parameters
    ----------
    G:
        the linear transformation matrix that links the visibilities to
        uniformly sampled sinusoids
    a_ri: 
        the visibility measurements
    K: 
        number of Diracs
    M: 
        the Fourier series expansion is between -M and M
    noise_level: 
        level of noise (ell_2 norm) in the measurements
    max_ini: 
        maximum number of initialisations
    stop_cri: 
        stopping criterion, either 'mse' or 'max_iter'
    """
    num_bands = a_ri.shape[1]  # number of bands considered
    L = 2 * M + 1  # length of the (complex-valued) b vector for each band
    a_ri = a_ri.flatten('F')
    # size of G: (Q(Q-1)) x (2M + 1), where Q is the number of antennas
    assert not np.iscomplexobj(G)  # G should be real-valued
    Gt_a = np.dot(G.T, a_ri)
    GtG = np.dot(G.T, G)
    # maximum number of iterations with each initialisation
    max_iter = 50

    # the least-square solution
    beta_ri = np.reshape(linalg.lstsq(G, a_ri)[0], (-1, num_bands), order='F')
    D1, D2 = hermitian_expan(M + 1)
    D = linalg.block_diag(D1, D2)
    D_coef1, D_coef2 = coef_expan_mtx(K)
    D_coef = linalg.block_diag(D_coef1, D_coef2)

    # shrink the output size to half as both the annihilating filter coeffiicnets and
    # the uniform samples of sinusoids are Hermitian symmetric
    mtx_shrink = output_shrink(K, L)

    # size of Tbeta_ri: (L - K)num_bands x 2(K + 1)
    Tbeta_ri = np.vstack([Tmtx_ri_half_out_half(beta_ri[:, band_count],
                                                K, D, L, D_coef, mtx_shrink)
                          for band_count in range(num_bands)])

    # size of various matrices / vectors
    sz_G1 = L * num_bands

    sz_Tb0 = (L - K) * num_bands  # the only effective size because of Hermitian symmetry

    sz_Rc0 = (L - K) * num_bands

    sz_coef = K + 1

    rhs = np.append(np.zeros(sz_coef + sz_Tb0 + sz_G1, dtype=float), 1)
    rhs_bl = np.concatenate((Gt_a, np.zeros(sz_Rc0, dtype=float)))

    # the main iteration with different random initialisations
    partial_dirac_recon = partial(dirac_recon_ri_multiband_inner,
                                  a_ri=a_ri, num_bands=num_bands, rhs=rhs,
                                  rhs_bl=rhs_bl, K=K, M=M, D1=D1, D2=D2,
                                  D_coef=D_coef, mtx_shrink=mtx_shrink,
                                  Tbeta_ri=Tbeta_ri, G=G, GtG=GtG, max_iter=max_iter)

    # generate all the random initialisations
    c_ri_half_all = np.random.randn(sz_coef, max_ini)

    res_all = []
    for loop in range(max_ini):
        res_all.append(
            partial_dirac_recon(c_ri_half_all[:, loop][:, np.newaxis])
        )

    # find the one with smallest error
    min_idx = np.array(zip(*res_all)[1]).argmin()
    c_opt, min_error, b_opt = res_all[min_idx]

    return c_opt, min_error, b_opt


def dirac_recon_ri_multiband_inner(c_ri_half, a_ri, num_bands, rhs, rhs_bl, K, M,
                                   D1, D2, D_coef, mtx_shrink, Tbeta_ri,
                                   G, GtG, max_iter):
    ''' Inner loop of the `dirac_recon_ri_multiband` function '''
    min_error = float('inf')
    # size of various matrices / vectors
    L = 2 * M + 1  # length of the (complex-valued) b vector

    sz_Tb0 = (L - K) * num_bands  # the only effective size because of Hermitian symmetry
    sz_Tb1 = K + 1

    sz_Rc0 = (L - K) * num_bands
    sz_Rc1 = L * num_bands

    sz_coef = K + 1
    sz_bri = L * num_bands

    # indices where the 4 x 4 block matrix is updated at each iteration
    # for -R(c)
    row_s1, row_e1 = sz_coef, sz_coef + sz_Tb0
    col_s1, col_e1 = sz_Tb0 + sz_Tb1, sz_Tb0 + sz_Tb1 + sz_Rc1
    # for -R(c).T
    row_s2, row_e2 = sz_coef + sz_Tb0, sz_coef + sz_Tb0 + sz_Rc1
    col_s2, col_e2 = sz_coef, sz_coef + sz_Tb0

    # GtG = np.dot(G.T, G)
    D = linalg.block_diag(D1, D2)

    c0_ri_half = c_ri_half.copy()
    error_seq = np.zeros(max_iter, dtype=float)

    Rmtx_band = Rmtx_ri_half_out_half(c_ri_half, K, D, L, D_coef, mtx_shrink)
    R_loop = linalg.block_diag(*([Rmtx_band] * num_bands))

    # first row of mtx_loop
    mtx_loop_first_row = np.hstack((np.zeros((sz_coef, sz_coef)), Tbeta_ri.T,
                                    np.zeros((sz_coef, sz_Rc1)), c0_ri_half))
    # last row of mtx_loop
    mtx_loop_last_row = np.hstack((c0_ri_half.T, np.zeros((1, sz_Tb0 + sz_Rc1 + 1))))

    # initialise mtx_loop and mtx_brecon
    mtx_loop = np.vstack((mtx_loop_first_row,
                          np.hstack((Tbeta_ri,
                                     np.zeros((sz_Tb0, sz_Tb0)),
                                     -R_loop,
                                     np.zeros((sz_Rc0, 1))
                                     )),
                          np.hstack((np.zeros((sz_Rc1, sz_Tb1)),
                                     -R_loop.T,
                                     GtG,
                                     np.zeros((sz_Rc1, 1))
                                     )),
                          mtx_loop_last_row
                          ))

    mtx_brecon = np.vstack((np.hstack((GtG, R_loop.T)),
                            np.hstack((R_loop, np.zeros((sz_Rc0, sz_Rc0))))
                            ))

    for inner in range(max_iter):
        # update the mtx_loop matrix
        mtx_loop[row_s1:row_e1, col_s1:col_e1] = -R_loop
        mtx_loop[row_s2:row_e2, col_s2:col_e2] = -R_loop.T

        # matrix should be symmetric
        mtx_loop += mtx_loop.T
        mtx_loop *= 0.5
        c_ri_half = linalg.solve(mtx_loop, rhs, check_finite=False)[:sz_coef]

        Rmtx_band = Rmtx_ri_half_out_half(c_ri_half, K, D, L, D_coef, mtx_shrink)
        R_loop = linalg.block_diag(*([Rmtx_band] * num_bands))

        # update mtx_brecon
        mtx_brecon[:sz_Rc1, sz_Rc1:] = R_loop.T
        mtx_brecon[sz_Rc1:, :sz_Rc1] = R_loop

        mtx_brecon += mtx_brecon.T
        mtx_brecon *= 0.5
        b_recon_ri = linalg.solve(mtx_brecon, rhs_bl, check_finite=False)[:sz_bri]

        error_seq[inner] = linalg.norm(a_ri - np.dot(G, b_recon_ri))
        if error_seq[inner] < min_error:
            min_error = error_seq[inner]
            b_recon_ri = np.reshape(b_recon_ri, (-1, num_bands), order='F')
            b_opt = np.dot(D1, b_recon_ri[:M + 1, :]) + \
                    1j * np.dot(D2, b_recon_ri[M + 1:, :])
            c_ri = np.dot(D_coef, c_ri_half)
            c_opt = c_ri[:K + 1] + 1j * c_ri[K + 1:]  # real and imaginary parts
    return c_opt, min_error, b_opt


def dirac_recon_ri_half_parallel(G, a_ri, K, M, max_ini=100):
    """
    Reconstruct point sources' locations (azimuth) from the visibility measurements.
    Here we enforce hermitian symmetry in the annihilating filter coefficients so that
    roots on the unit circle are encouraged.
    We use parallel implementation when stop_cri == 'max_iter'

    Parameters
    ----------
    G: 
        the linear transformation matrix that links the visibilities to
        uniformly sampled sinusoids
    a_ri: 
        the visibility measurements
    K: 
        number of Diracs
    M: 
        the Fourier series expansion is between -M and M
    noise_level: 
        level of noise (ell_2 norm) in the measurements
    max_ini: 
        maximum number of initialisations
    stop_cri: 
        stopping criterion, either 'mse' or 'max_iter'
    """
    L = 2 * M + 1  # length of the (complex-valued) b vector
    a_ri = a_ri.flatten('F')
    # size of G: (Q(Q-1)) x (2M + 1), where Q is the number of antennas
    assert not np.iscomplexobj(G)  # G should be real-valued
    Gt_a = np.dot(G.T, a_ri)
    # maximum number of iterations with each initialisation
    max_iter = 50

    # the least-square solution
    beta_ri = linalg.lstsq(G, a_ri)[0]
    D1, D2 = hermitian_expan(M + 1)
    D = linalg.block_diag(D1, D2)
    D_coef1, D_coef2 = coef_expan_mtx(K)
    D_coef = linalg.block_diag(D_coef1, D_coef2)

    # shrink the output size to half as both the annihilating filter coeffiicnets and
    # the uniform samples of sinusoids are Hermitian symmetric
    mtx_shrink = output_shrink(K, L)

    # size of Tbeta_ri: (L - K) x 2(K + 1)
    Tbeta_ri = Tmtx_ri_half_out_half(beta_ri, K, D, L, D_coef, mtx_shrink)

    # size of various matrices / vectors
    sz_G1 = L

    sz_Tb0 = L - K  # the only effective size because of Hermitian symmetry

    sz_Rc0 = L - K

    sz_coef = K + 1

    rhs = np.append(np.zeros(sz_coef + sz_Tb0 + sz_G1, dtype=float), 1)
    rhs_bl = np.concatenate((Gt_a, np.zeros(sz_Rc0, dtype=Gt_a.dtype)))

    # the main iteration with different random initialisations
    partial_dirac_recon = partial(dirac_recon_ri_inner, a_ri=a_ri, rhs=rhs,
                                  rhs_bl=rhs_bl, K=K, M=M, D1=D1, D2=D2,
                                  D_coef=D_coef, mtx_shrink=mtx_shrink,
                                  Tbeta_ri=Tbeta_ri, G=G, max_iter=max_iter)

    # generate all the random initialisations
    c_ri_half_all = np.random.randn(sz_coef, max_ini)

    res_all = []
    for loop in range(max_ini):
        res_all.append(
                partial_dirac_recon(c_ri_half_all[:,loop][:,np.newaxis])
                )

    # find the one with smallest error
    min_idx = np.array(zip(*res_all)[1]).argmin()
    c_opt, min_error, b_opt = res_all[min_idx]

    return c_opt, min_error, b_opt


def dirac_recon_ri_inner(c_ri_half, a_ri, rhs, rhs_bl, K, M,
                         D1, D2, D_coef, mtx_shrink, Tbeta_ri, G, max_iter):
    ''' inner loop of the `dirac_recon_ri_half_parallel` function '''

    min_error = float('inf')
    # size of various matrices / vectors
    L = 2 * M + 1  # length of the (complex-valued) b vector

    sz_Tb0 = L - K  # the only effective size because of Hermitian symmetry
    sz_Tb1 = K + 1

    sz_Rc0 = L - K
    sz_Rc1 = L

    sz_coef = K + 1
    sz_bri = L

    # indices where the 4 x 4 block matrix is updated at each iteration
    # for -R(c)
    row_s1, row_e1 = sz_coef, sz_coef + sz_Tb0
    col_s1, col_e1 = sz_Tb0 + sz_Tb1, sz_Tb0 + sz_Tb1 + sz_Rc1
    # for -R(c).T
    row_s2, row_e2 = sz_coef + sz_Tb0, sz_coef + sz_Tb0 + sz_Rc1
    col_s2, col_e2 = sz_coef, sz_coef + sz_Tb0

    GtG = np.dot(G.T, G)
    D = linalg.block_diag(D1, D2)

    c0_ri_half = c_ri_half.copy()
    error_seq = np.zeros(max_iter, dtype=float)
    R_loop = Rmtx_ri_half_out_half(c_ri_half, K, D, L, D_coef, mtx_shrink)

    # first row of mtx_loop
    mtx_loop_first_row = np.hstack((np.zeros((sz_coef, sz_coef)), Tbeta_ri.T,
                                    np.zeros((sz_coef, sz_Rc1)), c0_ri_half))
    # last row of mtx_loop
    mtx_loop_last_row = np.hstack((c0_ri_half.T, np.zeros((1, sz_Tb0 + sz_Rc1 + 1))))

    mtx_loop = np.vstack((mtx_loop_first_row,
                          np.hstack((Tbeta_ri,
                                     np.zeros((sz_Tb0, sz_Tb0)),
                                     -R_loop,
                                     np.zeros((sz_Rc0, 1))
                                     )),
                          np.hstack((np.zeros((sz_Rc1, sz_Tb1)),
                                     -R_loop.T,
                                     GtG,
                                     np.zeros((sz_Rc1, 1))
                                     )),
                          mtx_loop_last_row
                          ))

    mtx_brecon = np.zeros((sz_Rc1 + sz_Rc0, sz_Rc1 + sz_Rc0))
    mtx_brecon[:sz_Rc1, :sz_Rc1] = GtG

    for inner in range(max_iter):

        # update the mtx_loop matrix
        mtx_loop[row_s1:row_e1, col_s1:col_e1] = -R_loop
        mtx_loop[row_s2:row_e2, col_s2:col_e2] = -R_loop.T

        # matrix should be symmetric
        # mtx_loop = (mtx_loop + mtx_loop.T) / 2.
        mtx_loop += mtx_loop.T
        mtx_loop *= 0.5
        # c_ri_half = linalg.lstsq(mtx_loop, rhs)[0][:sz_coef]
        c_ri_half = linalg.solve(mtx_loop, rhs, check_finite=False)[:sz_coef]

        R_loop = Rmtx_ri_half_out_half(c_ri_half, K, D, L, D_coef, mtx_shrink)

        mtx_brecon[:sz_Rc1, sz_Rc1:] = R_loop.T
        mtx_brecon[sz_Rc1:, :sz_Rc1] = R_loop
        # mtx_brecon = (mtx_brecon + mtx_brecon.T) / 2.
        mtx_brecon += mtx_brecon.T
        mtx_brecon *= 0.5

        # b_recon_ri = linalg.lstsq(mtx_brecon, rhs_bl)[0][:sz_bri]
        b_recon_ri = linalg.solve(mtx_brecon, rhs_bl, check_finite=False)[:sz_bri]

        error_seq[inner] = linalg.norm(a_ri - np.dot(G, b_recon_ri))
        if error_seq[inner] < min_error:
            min_error = error_seq[inner]
            b_opt = np.dot(D1, b_recon_ri[:M + 1]) + 1j * np.dot(D2, b_recon_ri[M + 1:])
            c_ri = np.dot(D_coef, c_ri_half)
            c_opt = c_ri[:K + 1] + 1j * c_ri[K + 1:]  # real and imaginary parts
    return c_opt, min_error, b_opt

def make_G(p_mic_x, p_mic_y, omega_bands, sound_speed, M, signal_type='visibility'):
    """
    reconstruct point sources on the circle from the visibility measurements
    from multi-bands.

    Parameters
    ----------
    p_mic_x: 
        a vector that contains microphones' x-coordinates
    p_mic_y: 
        a vector that contains microphones' y-coordinates
    omega_bands: 
        mid-band (ANGULAR) frequencies [radian/sec]
    sound_speed: 
        speed of sound
    signal_type: 
        The type of the signal a, possible values are 'visibility' for
        covariance matrix and 'raw' for microphone inputs

    Returns
    -------
    The list of mapping matrices from measurements to sinusoids
    """
    
    # expansion matrices to take Hermitian symmetry into account
    norm_factor = np.reshape(sound_speed / omega_bands, (1, -1), order='F')
    # normalised antenna coordinates in a matrix form
    # each column corresponds to the location in one subband,
    # i.e., the second dimension corresponds to different subbands
    p_mic_x_normalised = np.reshape(p_mic_x, (-1, 1), order='F') / norm_factor
    p_mic_y_normalised = np.reshape(p_mic_y, (-1, 1), order='F') / norm_factor

    D1, D2 = hermitian_expan(M + 1)
    # G = mtx_fri2visi_ri_multiband(M, p_mic_x_normalised, p_mic_y_normalised, D1, D2)
    G_lst = mtx_fri2signal_ri_multiband(M, p_mic_x_normalised, p_mic_y_normalised,
                                        D1, D2, aslist=True, signal=signal_type)

    return np.array(G_lst)


def make_GtG_and_inv(G_lst):

    GtG_lst = []
    GtG_inv_lst = []
    for loop in range(len(G_lst)):
        G_loop = G_lst[loop]

        GtG = np.dot(G_loop.T, G_loop)


        GtG_lst.append(GtG)
        GtG_inv_lst.append(linalg.inv(GtG))

    return np.array(GtG_lst), np.array(GtG_inv_lst)


def pt_src_recon_multiband(a, p_mic_x, p_mic_y, omega_bands, sound_speed,
                           K, M, noise_level, max_ini=50,
                           update_G=False, verbose=False, signal_type='visibility', 
                           max_iter=50, 
                           G_lst=None, GtG_lst=None, GtG_inv_lst=None, 
                           **kwargs):
    """
    reconstruct point sources on the circle from the visibility measurements
    from multi-bands.

    Parameters
    ----------
    a: 
        the measured visibilities in a matrix form, where the second dimension
        corresponds to different subbands
    p_mic_x: 
        a vector that contains microphones' x-coordinates
    p_mic_y: 
        a vector that contains microphones' y-coordinates
    omega_bands: 
        mid-band (ANGULAR) frequencies [radian/sec]
    sound_speed: 
        speed of sound
    K: 
        number of point sources
    M: 
        the Fourier series expansion is between -M to M
    noise_level: 
        noise level in the measured visibilities
    max_ini: 
        maximum number of random initialisation used
    update_G: 
        update the linear mapping that links the uniformly sampled sinusoids to
        the visibility or not.
    verbose: 
        whether output intermediate results for debugging or not
    signal_type: 
        The type of the signal a, possible values are 'visibility' for
        covariance matrix and 'raw' for microphone inputs
    kwargs: 
        possible optional input: G_iter: number of iterations for the G updates
    """

    p_mic_x = np.squeeze(p_mic_x)
    p_mic_y = np.squeeze(p_mic_y)

    num_bands = np.array(omega_bands).size  # number of bands considered
    num_mic = p_mic_x.size

    assert M >= K
    assert num_mic * (num_mic - 1) >= 2 * M + 1
    if len(a.shape) == 2:
        assert a.shape[1] == num_bands

    a_ri = np.row_stack((a.real, a.imag))

    if update_G:
        if 'G_iter' in kwargs:
            max_loop_G = kwargs['G_iter']
        else:
            max_loop_G = 2
    else:
        max_loop_G = 1

    # initialisation
    min_error = float('inf')
    phik_opt = np.zeros(K)
    alphak_opt = np.zeros(K)

    # expansion matrices to take Hermitian symmetry into account
    norm_factor = np.reshape(sound_speed / omega_bands, (1, -1), order='F')
    # normalised antenna coordinates in a matrix form
    # each column corresponds to the location in one subband,
    # i.e., the second dimension corresponds to different subbands
    p_mic_x_normalised = np.reshape(p_mic_x, (-1, 1), order='F') / norm_factor
    p_mic_y_normalised = np.reshape(p_mic_y, (-1, 1), order='F') / norm_factor

    D1, D2 = hermitian_expan(M + 1)

    # Create G if necessary
    if G_lst is None:
        G_lst = make_G(p_mic_x, p_mic_y, omega_bands, sound_speed, M, signal_type=signal_type)
        GtG_lst, GtG_inv_lst = make_GtG_and_inv(G_lst)

    if GtG_lst is None or GtG_inv_lst is None:
        GtG_lst, GtG_inv_lst = make_GtG_and_inv(G_lst)

    for loop_G in range(max_loop_G):
        # c_recon, error_recon = \
        #     dirac_recon_ri_half_multiband_parallel(G, a_ri, K, M, max_ini)[:2]

        # tic = time.time()
        '''the original implementation'''
        # c_recon, error_recon = \
        #     dirac_recon_ri_half_multiband(G_lst, a_ri, K, M, max_ini)[:2]

        '''faster version with lu decomposition'''
        c_recon, error_recon = \
            dirac_recon_ri_half_multiband_lu(G_lst, GtG_lst, GtG_inv_lst,
                    a_ri, K, M, max_ini, max_iter=max_iter)[:2]
        # toc = time.time()
        # print(toc - tic)

        if verbose:
            print('noise level: {0:.3e}'.format(noise_level))
            print('objective function value: {0:.3e}'.format(error_recon))
        # recover Dirac locations
        uk = np.roots(np.squeeze(c_recon))
        uk /= np.abs(uk)
        phik_recon = np.mod(-np.angle(uk), 2. * np.pi)

        # use least square to reconstruct amplitudes
        if signal_type == 'visibility':
            error_loop = 0
            alphak_recon = []
            for band_count in range(num_bands):
                amp_mtx_ri_loop = build_mtx_amp_ri(
                    p_mic_x_normalised[:, band_count],
                    p_mic_y_normalised[:, band_count],
                    phik_recon
                )

                amplitude_band = \
                    sp.optimize.nnls(
                        np.dot(amp_mtx_ri_loop.T, amp_mtx_ri_loop),
                        np.dot(amp_mtx_ri_loop.T, a_ri[:, band_count])
                    )[0]

                error_loop += linalg.norm(a_ri[:, band_count] -
                                          np.dot(amp_mtx_ri_loop, amplitude_band)
                                          )

                alphak_recon.append(amplitude_band)

            alphak_recon = np.concatenate(alphak_recon)

        elif signal_type == 'raw':
            partial_build_mtx_amp = partial(build_mtx_raw_amp, phi_k=phik_recon)
            blocks = []
            for band_count in range(num_bands):
                amp_loop = partial_build_mtx_amp(
                        p_mic_x_normalised[:,band_count],
                        p_mic_y_normalised[:,band_count],
                        )
                blocks.append(amp_loop)
            amp_mtx = linalg.block_diag(blocks)


            alphak_recon = sp.linalg.lstsq(amp_mtx, a.flatten('F'))[0]

            error_loop = linalg.norm(a.flatten('F') - np.dot(amp_mtx, alphak_recon))

        else:
            raise ValueError('signal_type must be ''raw'' or ''visibility''.')

        if error_loop < min_error:
            min_error = error_loop
            phik_opt = phik_recon
            alphak_opt = np.reshape(alphak_recon, (-1, num_bands), order='F')

        # update the linear transformation matrix
        if update_G:
            G_lst = mtx_updated_G_multiband_new(phik_opt, M, p_mic_x_normalised,
                                                p_mic_y_normalised, G_lst, num_bands)
            GtG_lst, GtG_inv_lst = make_GtG_and_inv(G_lst)

    # convert propagation vector to DOA
    phik_doa = np.mod(phik_opt - np.pi, 2. * np.pi)
    return phik_doa, alphak_opt


def pt_src_recon(a, p_mic_x, p_mic_y, omega_band, sound_speed,
                 K, M, noise_level, max_ini=50,
                 stop_cri='mse', update_G=False, verbose=False, signal_type='visibility', **kwargs):
    """
    reconstruct point sources on the circle from the visibility measurements

    Parameters
    ----------
    a: 
        the measured visibilities
    p_mic_x: 
        a vector that contains microphones' x-coordinates
    p_mic_y: 
        a vector that contains microphones' y-coordinates
    omega_band: 
        mid-band (ANGULAR) frequency [radian/sec]
    sound_speed: 
        speed of sound
    K: 
        number of point sources
    M: 
        the Fourier series expansion is between -M to M
    noise_level: 
        noise level in the measured visibilities
    max_ini: 
        maximum number of random initialisation used
    stop_cri: 
        either 'mse' or 'max_iter'
    update_G: 
        update the linear mapping that links the uniformly sampled sinusoids to
        the visibility or not.
    verbose: 
        whether output intermediate results for debugging or not
    signal_type: 
        The type of the signal a, possible values are 'visibility' for
        covariance matrix and 'raw' for microphone inputs
    kwargs: possible optional input: G_iter: number of iterations for the G updates
    """
    assert M >= K
    num_mic = p_mic_x.size
    assert num_mic * (num_mic - 1) >= 2 * M + 1
    a_ri = np.concatenate((a.real, a.imag))
    if update_G:
        if 'G_iter' in kwargs:
            max_loop_G = kwargs['G_iter']
        else:
            max_loop_G = 2
    else:
        max_loop_G = 1

    # initialisation
    min_error = float('inf')
    phik_opt = np.zeros(K)
    alphak_opt = np.zeros(K)

    # expansion matrices to take Hermitian symmetry into account
    p_mic_x_normalised = p_mic_x / (sound_speed / omega_band)
    p_mic_y_normalised = p_mic_y / (sound_speed / omega_band)
    D1, D2 = hermitian_expan(M + 1)
    G = mtx_fri2signal_ri(M, p_mic_x_normalised, p_mic_y_normalised, D1, D2, signal=signal_type)
    for loop_G in range(max_loop_G):
        # c_recon, error_recon = dirac_recon_ri(G, a_ri, K_est, M, noise_level, max_ini, stop_cri)[:2]
        # c_recon, error_recon = dirac_recon_ri_half(G, a_ri, K, M, noise_level, max_ini, stop_cri)[:2]
        c_recon, error_recon = dirac_recon_ri_half_parallel(G, a_ri, K, M, max_ini)[:2]

        if verbose:
            print('noise level: {0:.3e}'.format(noise_level))
            print('objective function value: {0:.3e}'.format(error_recon))
        # recover Dirac locations
        uk = np.roots(np.squeeze(c_recon))
        uk /= np.abs(uk)
        phik_recon = np.mod(-np.angle(uk), 2. * np.pi)

        # use least square to reconstruct amplitudes
        amp_mtx = build_mtx_amp(phik_recon, p_mic_x_normalised, p_mic_y_normalised)
        amp_mtx_ri = np.vstack((amp_mtx.real, amp_mtx.imag))
        alphak_recon = sp.optimize.nnls(amp_mtx_ri, a_ri.squeeze())[0]
        error_loop = linalg.norm(a.flatten('F') - np.dot(amp_mtx, alphak_recon))

        if error_loop < min_error:
            min_error = error_loop
            # amp_sort_idx = np.argsort(alphak_recon)[-K:]
            # phik_opt, alphak_opt = phik_recon[amp_sort_idx], alphak_recon[amp_sort_idx]
            phik_opt, alphak_opt = phik_recon, alphak_recon

        # update the linear transformation matrix
        if update_G:
            G = mtx_updated_G(phik_recon, M, amp_mtx_ri, G)

    return phik_opt, alphak_opt


def pt_src_recon_rotate(a, p_mic_x, p_mic_y, K, M, noise_level, max_ini=50,
                        stop_cri='mse', update_G=False, num_rotation=1,
                        verbose=False, signal_type='visibility', **kwargs):
    """
    reconstruct point sources on the circle from the visibility measurements.
    Here we apply random rotations to the coordiantes.

    Parameters
    ----------
    a: 
        the measured visibilities
    p_mic_x: 
        a vector that contains microphones' x-coordinates
    p_mic_y: 
        a vector that contains microphones' y-coordinates
    K: 
        number of point sources
    M: 
        the Fourier series expansion is between -M to M
    noise_level: 
        noise level in the measured visibilities
    max_ini: 
        maximum number of random initialisation used
    stop_cri: 
        either 'mse' or 'max_iter'
    update_G: 
        update the linear mapping that links the uniformly sampled sinusoids to
        the visibility or not.
    num_rotation: 
        number of random rotations
    verbose: 
        whether output intermediate results for debugging or not
    signal_type: 
        The type of the signal a, possible values are 'visibility' for
        covariance matrix and 'raw' for microphone inputs
    kwargs: 
        possible optional input: G_iter: number of iterations for the G updates
    """
    assert M >= K
    num_mic = p_mic_x.size
    assert num_mic * (num_mic - 1) >= 2 * M + 1
    a_ri = np.concatenate((a.real, a.imag))
    if update_G:
        if 'G_iter' in kwargs:
            max_loop_G = kwargs['G_iter']
        else:
            max_loop_G = 2
    else:
        max_loop_G = 1

    # expansion matrices to take Hermitian symmetry into account
    D1, D2 = hermitian_expan(M + 1)

    # initialisation
    min_error = float('inf')
    phik_opt = np.zeros(K)
    alphak_opt = np.zeros(K)

    # random rotation angles
    rotate_angle_all = np.random.rand(num_rotation) * np.pi * 2.
    for rand_rotate in range(num_rotation):
        rotate_angle_loop = rotate_angle_all[rand_rotate]
        rotate_mtx = np.array([[np.cos(rotate_angle_loop), -np.sin(rotate_angle_loop)],
                               [np.sin(rotate_angle_loop), np.cos(rotate_angle_loop)]])

        # rotate microphone steering vector
        # (due to change of coordinate w.r.t. the random rotation)
        p_mic_xy_rotated = np.dot(rotate_mtx,
                                  np.row_stack((p_mic_x.flatten('F'),
                                                p_mic_y.flatten('F')))
                                  )
        p_mic_x_rotated = np.reshape(p_mic_xy_rotated[0, :], p_mic_x.shape, order='F')
        p_mic_y_rotated = np.reshape(p_mic_xy_rotated[1, :], p_mic_y.shape, order='F')

        # linear transformation matrix that maps uniform samples
        # of sinusoids to visibilities
        G = mtx_fri2signal_ri(M, p_mic_x_rotated, p_mic_y_rotated, D1, D2, signal=signal_type)

        for loop_G in range(max_loop_G):
            c_recon, error_recon = dirac_recon_ri_half(G, a_ri, K, M, noise_level,
                                                       max_ini, stop_cri)[:2]
            if verbose:
                print('noise level: {0:.3e}'.format(noise_level))
                print('objective function value: {0:.3e}'.format(error_recon))

            # recover Dirac locations
            uk = np.roots(np.squeeze(c_recon))
            uk /= np.abs(uk)
            phik_recon_rotated = np.mod(-np.angle(uk), 2. * np.pi)

            # use least square to reconstruct amplitudes
            amp_mtx = build_mtx_amp(phik_recon_rotated, p_mic_x_rotated, p_mic_y_rotated)
            amp_mtx_ri = np.vstack((amp_mtx.real, amp_mtx.imag))
            alphak_recon = sp.optimize.nnls(amp_mtx_ri, a_ri.squeeze())[0]
            error_loop = linalg.norm(a.flatten('F') - np.dot(amp_mtx, alphak_recon))

            if error_loop < min_error:
                min_error = error_loop

                # rotate back
                # transform to cartesian coordinate
                xk_recon_rotated, yk_recon_rotated = polar2cart(1, phik_recon_rotated)
                xy_rotate_back = linalg.solve(rotate_mtx,
                                              np.row_stack((xk_recon_rotated.flatten('F'),
                                                            yk_recon_rotated.flatten('F')))
                                              )
                xk_recon = np.reshape(xy_rotate_back[0, :], xk_recon_rotated.shape, 'F')
                yk_recon = np.reshape(xy_rotate_back[1, :], yk_recon_rotated.shape, 'F')
                # transform back to polar coordinate
                phik_recon = np.mod(np.arctan2(yk_recon, xk_recon), 2. * np.pi)

                phik_opt, alphak_opt = phik_recon, alphak_recon

            # update the linear transformation matrix
            if update_G:
                G = mtx_updated_G(phik_recon_rotated, M, amp_mtx_ri, G)

    return phik_opt, alphak_opt
