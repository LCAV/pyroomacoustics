import numpy as np
from ..transform import STFT, compute_synthesis_window

def griffin_lim(
    X,
    hop,
    analysis_window,
    fft_size=None,
    stft_kwargs={},
    n_iter=100,
    ini=None,
    callback=None,
):
    """
    Implementation of the Griffin-Lim phase reconstruction algorithm from STFT magnitude measurements.

    Parameters
    ----------
    X: array_like, shape (n_frames, n_freq)
        The STFT magnitude measurements
    stft_kwargs: dict
        Dictionary of parameters for the STFT
    ini: str or array_like, np.complex, shape (n_frames, n_freq)
        The initial value of the phase estimate. If "random", uses a random guess. If ``None``, uses ``0`` phase.
    n_iter: int
        The number of iteration
    callback: func
        A callable taking as argument an int and the reconstructed STFT and time-domain signals
    """

    if isinstance(ini, str) and ini == "random":
        ini = np.exp(1j * 2 * np.pi * np.random.rand(*X.shape))
    elif ini is None:
        ini = np.ones(X.shape, dtype=np.complex128)

    # take care of the STFT parameters
    if fft_size is None:
        fft_size = 2 * (X.shape[1] - 1)

    # the optimal GL window
    synthesis_window = compute_synthesis_window(analysis_window, hop)

    # create the STFT object
    engine = STFT(
        fft_size,
        hop=hop,
        analysis_window=analysis_window,
        synthesis_window=synthesis_window,
        **stft_kwargs
    )

    # Initialize the signal
    Y = X * ini
    y = engine.synthesis(Y)

    # the successive application of analysis/synthesis introduces
    # a shift of ``fft_size - hop`` that we must correct
    the_shift = fft_size - hop
    y[:-the_shift,] = y[the_shift:,]

    for epoch in range(n_iter):

        # possibly monitor the reconstruction
        if callback is not None:
            callback(epoch, Y, y)

        # back to STFT domain
        Y[:, :] = engine.analysis(y)

        # enforce magnitudes
        Y *= X / np.abs(Y)

        # back to time domain
        y[:-the_shift,] = engine.synthesis(Y)[the_shift:,]

    # last callback
    if callback is not None:
        callback(epoch, Y, y)

    return y
