
import numpy as np
from scipy.signal import fftconvolve

def trinicon(signals, w0=None,
        filter_length=2048,
        block_length=None, 
        n_blocks=8,
        alpha_on=None,
        j_max=10,
        delta_max=1e-4,
        sigma2_0=1e-7,
        mu=0.001,
        lambd_a=0.2,
        return_filters=False):
    '''
    Implementation of the TRINICON Blind Source Separation algorithm as described in

    R. Aichner, H. Buchner, F. Yan, and W. Kellermann  *A real-time
    blind source separation scheme and its application to reverberant and noisy
    acoustic environments*,  Signal Processing, 86(6), 1260-1277.
    doi:10.1016/j.sigpro.2005.06.022, 2006. `[pdf] <http://www.buchner-net.com/lnt2006_19.pdf>`_

    Specifically, adaptation of the pseudo-code from Table 1.

    The implementation is hard-coded for 2 output channels.

    Parameters
    ----------
    signals: ndarray (nchannels, nsamples)
        The microphone input signals (time domain)
    w0: ndarray (nchannels, nsources, nsamples), optional
        Optional initial value for the demixing filters
    filter_length: int, optional
        The length of the demixing filters, if w0 is provided, this option is ignored
    block_length: int, optional
        Block length (default 2x filter_length)
    n_blocks: int, optional
        Number of blocks processed at once (default 8)
    alpha_on: int, optional
        Online overlap factor (default ``n_blocks``)
    j_max: int, optional
        Number of offline iterations (default 10)
    delta_max: float, optional
        Regularization parameter, this sets the maximum value of the regularization term (default 1e-4)
    sigma2_0: float, optional
        Regularization parameter, this sets the reference (machine?) noise
        level in the regularization (default 1e-7)
    mu: float, optional
        Offline update step size (default 0.001)
    lambd_a: float, optional
        Online forgetting factor (default 0.2)
    return_filters: bool
        If true, the function will return the demixing matrix too (default False)

    Returns
    -------
    ndarray
        Returns an (nsources, nsamples) array. Also returns
        the demixing matrix (nchannels, nsources, nsamples)
        if ``return_filters`` keyword is True.
    '''

    P = signals.shape[0] # number of microphones
    Q = 2                # number of output channels


    # the filters
    if w0 is None:
        L = filter_length
        w = np.zeros((P,Q,L))
        w[:P//2,0,L//2] = 1
        w[P//2:,1,L//2] = 1
    else:
        w = w0.copy()
        L = w0.shape[2]

    K = n_blocks         # Number of successive blocks processed at the same time
    if block_length is None:  # Block length
        N = 2*L
    else:
        N = block_length

    if alpha_on is None:
        alpha_on = K         # online overlap factor

    hop = K * L // alpha_on

    # pad with zeros to have a whole number of online blocks
    if signals.shape[1] % hop != 0:
        signals = np.concatenate((signals, np.zeros((P, hop - (signals.shape[0]%hop)))), axis=1)

    S = signals.shape[1] # total signal length
    M = S / hop          # number of online blocks

    y = np.zeros((Q,S))    # the processed output signal

    m = 1               # online block index
    while m <= M:        # online loop
        
        # new chunk of input signal
        x = np.zeros((P,K*L+N))
        if m*hop > S:
            # we need some zero padding at the back
            le = S - (m-1)*hop + N
            x[:le] = signals[:,m*hop-K*L-N:]
        if m*hop >= K*L+N:
            x = signals[:,m*hop-K*L-N:m*hop]
        else:
            # we need some zero padding at the beginning
            x[:,-m*hop:] = signals[:,:m*hop]

        # use filter from previous iteration to initialize offline part
        w_new = w.copy()

        for j in range(j_max):     # offline update loop

            y_c = np.zeros((Q,K*L+N-L))  # c stands for chunk
            y_blocks = np.zeros((Q,K,N))

            for q in range(Q):
                # convolve with filters
                for p in range(P):
                    # We discard the 'oldest' output of the convolution according
                    # to the filter matrix definition (6) in the paper
                    y_c[q,:] += fftconvolve(x[p,:], w_new[p,q,:], mode='valid')[1:]

                # split into smaller blocks
                for i in range(K):
                    y_blocks[q,i,:] = y_c[q,i*L:i*L+N]

            # blocks energy
            sigma2 = np.sum(y_blocks**2, axis=2)

            # cross-correlations
            # XXX This is the part hard coded for two channels XXX
            r_cross = np.zeros((Q,K,2*L-1))
            for i in range(K):
                y0 = y_c[0,i*L:i*L+N]
                y1 = y_c[1,i*L:i*L+N]
                r = fftconvolve(y1, y0[::-1], mode='full')
                r_cross[0,i,:] = r[N-L:N+L-1]         # r_y1y0
                r_cross[1,i,:] = r_cross[0,i,::-1]    # r_y0y1 by symmetry is just r_y1y0 reversed

            # regularization term
            delta = delta_max*np.exp(-sigma2/sigma2_0)

            # offline update
            delta_w = np.zeros((P,Q,L))
            for q in range(Q):
                for p in range(P):
                    for i in range(K):
                        # this implements the row-wise sylvester constraint as explained in Fig. 4 (b) of paper
                        delta_w[p,q,:] += fftconvolve(r_cross[q,i,:]/(sigma2[q,i]+delta[q,i]), w_new[p,1-q,::-1], mode='valid')[::-1]
                    delta_w[p,q,:] /= K

            w_new = w_new - mu*delta_w

        # online update
        w = lambd_a*w + (1-lambd_a)*w_new

        # compute output signal
        for q in range(Q):
            for p in range(P):
                y[q,(m-1)*hop:m*hop] += fftconvolve(x[p,-hop-L+1:], w[p,q,:], mode='valid')
        
        # next block
        m += 1

    if return_filters:
        return y, w
    else:
        return y
