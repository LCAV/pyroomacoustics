
import numpy as np

def frac_delay(delta, N, w_max=0.9, C=4):
    '''
    Compute optimal fractionnal delay filter according to

    Design of Fractional Delay Filters Using Convex Optimization
    William Putnam and Julius Smith

    Parameters
    ----------
    delta: 
        delay of filter in (fractionnal) samples
    N: 
        number of taps
    w_max: 
        Bandwidth of the filter (in fraction of pi) (default 0.9)
    C: 
        sets the number of constraints to C*N (default 4)
    '''

    # constraints
    N_C = int(C*N)
    w = np.linspace(0, w_max*np.pi, N_C)[:,np.newaxis]
    
    n = np.arange(N)

    try:
        from cvxopt import solvers, matrix
    except:
        raise ValueError('To use the frac_delay function, the cvxopt module is necessary.')

    f = np.concatenate((np.zeros(N), np.ones(1)))

    A = []
    b = []
    for i in range(N_C):
        Anp = np.concatenate(([np.cos(w[i]*n), -np.sin(w[i]*n)], [[0],[0]]), axis=1)
        Anp = np.concatenate(([-f], Anp), axis=0)
        A.append(matrix(Anp))
        b.append(matrix(np.concatenate(([0], np.cos(w[i]*delta), -np.sin(w[i]*delta)))))

    solvers.options['show_progress'] = False
    sol = solvers.socp(matrix(f), Gq=A, hq=b)

    h = np.array(sol['x'])[:-1,0]

    '''
    import matplotlib.pyplot as plt
    w = np.linspace(0, np.pi, 2*N_C)
    F = np.exp(-1j*w[:,np.newaxis]*n)
    Hd = np.exp(-1j*delta*w)
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(np.abs(np.dot(F,h) - Hd))
    plt.subplot(3,1,2)
    plt.plot(np.diff(np.angle(np.dot(F,h))))
    plt.subplot(3,1,3)
    plt.plot(h)
    '''

    return h


def low_pass(numtaps, B, epsilon=0.1):

    bands = [0, (1-epsilon)*B, B, 0.5]
    desired = [1, 0]

    from scipy.signal import remez

    h = remez(numtaps, bands, desired, grid_density=32)

    '''
    import matplotlib.pyplot as plt
    w = np.linspace(0, np.pi, 8*numtaps)
    F = np.exp(-1j*w[:,np.newaxis]*np.arange(numtaps))
    Hd = np.exp(-1j*w)
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(np.abs(np.dot(F,h)))
    plt.subplot(3,1,2)
    plt.plot(np.angle(np.dot(F,h)))
    plt.subplot(3,1,3)
    plt.plot(h)
    '''
    
    return h

def resample(x, p, q):

    import fractions
    gcd = fractions.gcd(p,q)
    p /= gcd
    q /= gcd

    m = np.maximum(p,q)
    h = low_pass(10*m+1, 1./(2.*m))

    x_up = np.kron(x, np.concatenate(([1], np.zeros(p-1))))

    from scipy.signal import fftconvolve
    x_rs = fftconvolve(x_up, h)

    x_ds = x_rs[h.shape[0]/2+1::q]
    x_ds = x_ds[:np.floor(x.shape[0]*p/q)]

    '''
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(x)
    plt.subplot(3,1,2)
    plt.plot(x_rs)
    plt.subplot(3,1,3)
    plt.plot(x_ds)
    '''

    return x_ds

