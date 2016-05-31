
import numpy as np

def phat(x1, x2):

    N1 = x1.shape[0]
    N2 = x2.shape[0]

    N = N1 + N2 - 1

    X1 = np.fft.rfft(x1, n=N)
    X1 /= np.abs(X1)

    X2 = np.fft.rfft(x2, n=N)
    X2 /= np.abs(X2)

    r_12 = np.fft.irfft(X1*np.conj(X2), n=N)

    '''
    import matplotlib.pyplot as plt 
    plt.figure()
    plt.plot(r_12)
    plt.show()
    '''

    i = np.argmax(np.abs(r_12))

    if i < N1:
        return i
    else:
        return i - N1 - N2 + 1

def correlation(x1, x2):

    N1 = x1.shape[0]
    N2 = x2.shape[0]

    N = N1 + N2 - 1

    x1_p = np.zeros(N)
    x1_p[:N1] = x1
    x2_p = np.zeros(N)
    x2_p[:N2] = x2

    X1 = np.fft.fft(x1_p)

    X2 = np.fft.fft(x2_p)

    r_12 = np.real(np.fft.ifft(X1*np.conj(X2)))

    '''
    import matplotlib.pyplot as plt 
    plt.figure()
    plt.plot(np.real(r_12))
    plt.plot(np.imag(r_12))
    plt.show()
    '''

    i = np.argmax(r_12)

    if i < N1:
        return i
    else:
        return i - N1 - N2 + 1


def delay_estimation(x1, x2, L):
    '''
    Estimate the delay between x1 and x2.
    L is the block length used for phat
    '''

    K = np.minimum(x1.shape[0], x2.shape[0])/L

    delays = np.zeros(K)
    for k in range(K):
        delays[k] = phat(x1[k*L:(k+1)*L], x2[k*L:(k+1)*L])

    return int(np.median(delays))


def time_align(ref, deg, L=4096):
    '''
    return a copy of deg time-aligned and of same-length as ref.
    L is the block length used for correlations.
    '''

    # estimate delay of signal
    from numpy import zeros, minimum
    delay = delay_estimation(ref, deg, L)

    # time-align with reference segment for error metric computation
    sig = zeros(ref.shape[0])
    if (delay >= 0):
        length = minimum(deg.shape[0], ref.shape[0]-delay)
        sig[delay:length+delay] = deg[:length]
    else:
        length = minimum(deg.shape[0]+delay, ref.shape[0])
        sig = zeros(ref.shape)
        sig[:length] = deg[-delay:-delay+length]

    return sig

