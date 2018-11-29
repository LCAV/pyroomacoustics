import numpy as np
from scipy.signal import resample


def SpaRIR(G, S, delay=20, weights=np.array([]), q=1, gini=0):
    L = G.shape[0]
    print(G.shape)

    if np.mod(L, 2) == 1:
        print('The length of G must be even.')
        exit()

    #S = S[1:int(L / 2 + 1)]
    # S(k) and G(k) for k > L / 2 + 1 not used due to the symmetry of FFT (g is assumed to be real-valued)

    if q > 1:
        G = np.array([G[0:L / 2 + 1], np.zeros(q * L - L / 2 - 1, 1)])
        S = np.array([S, (q * L - L / 2 - 1, 1) == 0])
        L = q * L

    y = np.concatenate((np.real(G[S]), np.imag(G[S])),axis=0)
    print(y.shape)
    M = y.shape[0]


    delay = q * delay

    if gini == 0:  # if no initialization is given
        g = np.zeros((L, 1))
        g[delay + 1] = 1
    else:
        g = gini

    if weights == 0:
        tau = np.sqrt(L) / (y.T.dot(y))  # * ones(L,1)
        tau = tau * np.exp(0.11 * np.abs((np.arange(1., L + 1.).T - delay) / q) ** 0.3)
    elif weights.size == 0:
        tau = np.sqrt(L) / (y.T @ y)
        tau = tau * np.exp(0.11 * np.abs((np.arange(1., L + 1.).T - delay) / q) ** 0.3)
        tau = tau.T
        print(tau.shape)
    elif weights.shape[0] == 1:
        tau = np.ones((L, 1)) * weights
    else:
        tau = np.tile(weights.T, (q, 1)).reshape(L)

    maxiter = 10000
    alphamax = 1e5  # maximum step - length parameter alpha
    alphamin = 1e-7  # minimum step - length parameteralpha
    tol = 1e-4

    aux = np.zeros((L, 1))
    G = np.fft.fft(g)
    Ag = np.concatenate((np.real(G[S]), np.imag(G[S])),axis=0)
    r = Ag - y  # instead of r = A * g - y
    aux[S] = r[0:int(M / 2)] + 1j * r[int(M / 2):None]
    print('aux')
    print(aux.shape)
    gradq = L / 2 * np.fft.ifft(aux, L)  # instead of gradq = A'*r
    alpha = 10
    support = g != 0
    print('gradq')
    print(gradq.shape)
    iter = 0

    crit = np.zeros((maxiter, 1))
    criterion = -tau[support] * np.sign(g[support]) - gradq[support]
    crit[iter + 1] = np.sum(criterion ** 2)

    while (crit[iter + 1] > tol) and iter < maxiter:
        prev_r = r
        prev_g = g
        g = soft(prev_g - gradq * (1 / alpha), tau / alpha)
        dg = g - prev_g
        DG = np.fft.fft(dg)
        Adg = np.array([np.real(DG(S)), np.imag(DG(S))])
        r = prev_r + Adg  # faster than A * g - y
        dd = dg.flatten().T @ dg.flatten()
        dGd = Adg.flatten().T @ Adg.flatten()
        alpha = np.min(alphamax, np.max(alphamin, dGd / (np.finfo(np.float32).min + dd)))
        iter = iter + 1
        support = g != 0
        aux[S] = r[0: int(M / 2)] + 1j * r[int(M / 2): None]
        gradq = L / 2 * np.fft.ifft(aux, L)
        criterion = -tau[support] * np.sign(g[support]) - gradq[support]
        crit[iter + 1] = sum(criterion ** 2) + sum(abs(gradq[~support]) - tau[~support] > tol)

    if q > 1:
        g = q * resample(g, 1, q, 100)

    print('SpaRIR: {0} iterations done.'.format(iter))

    if sum(g == 0) == L:
        print('Zero solution! Do you really like it?')
        print('Check input parameters such as weights, delay, gini or')
        print('internal parameters such as tol, alphamax, alphamin, etc.')


def soft(x, T):
    if np.sum(np.abs(T)) == 0:
        y = x
    else:
        y = np.max(np.abs(x) - T, 0)
        # y = sign(x). * y
        y = y / (y + T) * x
    return y
