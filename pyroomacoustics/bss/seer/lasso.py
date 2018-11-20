import numpy as np

def lasso(W, S, mu):
    n_freq, n_chan, n_src = W.shape
    k_freq = S.shape[0]
    # Here comes Lassoooooooooo
    Z = np.zeros((n_chan, k_freq), dtype=W.dtype)
    Hrtf = np.zeros((n_freq, n_chan), dtype=W.dtype)
    for i in range(n_chan):
        Z[i, :] = np.array([W[S[f], 0, i] / W[S[f], 1, i] for f in range(k_freq)]).conj().T
        # Hrtf[:, i] = np.argmin(np.linalg.norm(Z[i,:] - ))
