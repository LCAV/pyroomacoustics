import numpy as np
from scipy.signal import resample


def SpaRIR(G, S, delay=0, weights=np.array([]), q=1, gini=0):
    L = G.shape[0]  # n_freq

    if q > 1:
        G = np.array([G[0:L / 2 + 1], np.zeros(q * L - L / 2 - 1, 1)])
        S = np.array([S, (q * L - L / 2 - 1, 1) == 0])
        L = q * L

    y = np.concatenate((np.real(G[S]), np.imag(G[S])), axis=0)
    M = y.shape[0]

    delay = q * delay

    if gini == 0:  # if no initialization is given
        g = np.zeros((L, 1))
        g[delay] = 1
    else:
        g = gini

    if weights == 0:
        tau = np.sqrt(L) / (y.conj().T.dot(y))  # * ones(L,1)
        tau = tau * np.exp(0.11 * np.abs((np.arange(1., L + 1.).T - delay) / q) ** 0.3)
    elif weights.size == 0:
        tau = np.sqrt(L) / (y.conj().T.dot(y))
        tau = tau * np.exp(0.11 * np.abs((np.arange(1., L + 1.).T - delay) / q) ** 0.3)
        tau = tau.T
    elif weights.shape[0] == 1:
        tau = np.ones((L, 1)) * weights
    else:
        tau = np.tile(weights.T, (q, 1)).reshape(L)

    maxiter = 50
    alphamax = 1e5  # maximum step - length parameter alpha
    alphamin = 1e-7  # minimum step - length parameteralpha
    tol = 10

    aux = np.zeros((L, 1),dtype=complex)
    G = np.fft.fft(g.flatten())
    Ag = np.concatenate((np.real(G[S]), np.imag(G[S])), axis=0)
    r = Ag - y.flatten()  # instead of r = A * g - y
    aux[S] = np.expand_dims(r[0:M // 2] + 1j * r[M // 2:], axis=1)
    gradq = L * np.fft.irfft(aux.flatten(), L)  # instead of gradq = A'*r
    gradq = np.expand_dims(gradq, axis=1)
    alpha = 10
    support = g != 0
    iter_ = 0

    crit = np.zeros((maxiter, 1))

    criterion = -tau[support] * np.sign(g[support]) - gradq[support]
    crit[iter_] = np.sum(criterion ** 2)
    # print("iteration: ", iter_ + 1, ", criterion: ", crit[iter_])

    while (crit[iter_] > tol) and (iter_ < maxiter - 1):
        prev_r = r
        prev_g = g
        g = soft(prev_g - gradq * (1 / alpha), tau / alpha)
        dg = g - prev_g
        DG = np.fft.fft(dg.flatten())
        Adg = np.concatenate((np.real(DG[S]), np.imag(DG[S])), axis=0)
        r = prev_r + Adg.flatten()  # faster than A * g - y
        dd = dg.flatten().conj().T @ dg.flatten()
        dGd = Adg.flatten().conj().T @ Adg.flatten()
        alpha = min(alphamax, max(alphamin, dGd / (np.finfo(np.float32).eps + dd)))
        iter_ = iter_ + 1
        support = g != 0
        aux[S] = np.expand_dims(r[0:M // 2] + 1j * r[M // 2:], axis=1)
        gradq = L * np.fft.irfft(aux.flatten(), L)
        gradq = np.expand_dims(gradq, axis=1)
        criterion = -tau[support] * np.sign(g[support]) - gradq[support]
        crit[iter_] = sum(criterion ** 2) + sum(abs(gradq[~support]) - tau[~support] > tol)
        # if iter_ % 100 == 0:
        # print("iteration: ", iter_+1, ", criterion: ", crit[iter_])

    if q > 1:
        g = q * resample(g, 1, q, 100)

    print('SpaRIR: {0} iterations done.'.format(iter_))

    return g.flatten()


def soft(x, T):
    if np.sum(np.abs(T).flatten()) == 0:
        y = x
    else:
        y = np.max(np.abs(x) - T, 0)
        # y = sign(x). * y
        y = y / (y + T) * x
    return y
