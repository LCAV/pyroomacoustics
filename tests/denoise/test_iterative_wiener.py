import numpy as np

import pyroomacoustics as pra


def test_iterative_wiener():
    """
    A simple functional test that the call does not produce any errors.
    """
    # parameters
    num_blocks = 20
    nfft = 512
    hop = nfft // 2

    # create a dummy signal
    blocks = np.random.randn(num_blocks, hop)

    # initialize STFT and IterativeWiener objects
    stft = pra.transform.STFT(nfft, hop=hop, analysis_window=pra.hann(nfft))
    scnr = pra.denoise.IterativeWiener(
        frame_len=nfft, lpc_order=20, iterations=2, alpha=0.8, thresh=0.01
    )

    # apply block-by-block
    processed_blocks = []
    for n in range(num_blocks):

        # go to frequency domain, 50% overlap
        stft.analysis(blocks[n])

        # compute wiener output
        X = scnr.compute_filtered_output(
            current_frame=stft.fft_in_buffer, frame_dft=stft.X
        )

        # back to time domain
        mono_denoised = stft.synthesis(X)
        processed_blocks.append(mono_denoised)


def test_iterative_wiener_output_scaling():
    """
    Regression test for https://github.com/LCAV/pyroomacoustics/issues/420.

    The iterative Wiener filter used to scale its output down by a factor of
    ``sqrt(frame_len)``, so the enhanced signal had a much lower amplitude than
    the input. For an almost noise-free, near-passthrough signal the output RMS
    should stay close to the input RMS (and definitely not collapse by
    ``1 / sqrt(nfft)``).
    """
    from pyroomacoustics.denoise.iterative_wiener import apply_iterative_wiener

    rng = np.random.RandomState(0)
    fs = 16000
    t = np.arange(fs) / fs
    # harmonic (speech-like) signal so most frames trigger the speech branch
    clean = (
        0.6 * np.sin(2 * np.pi * 220 * t)
        + 0.3 * np.sin(2 * np.pi * 440 * t)
        + 0.1 * np.sin(2 * np.pi * 880 * t)
    )
    noisy = clean + 0.01 * rng.randn(*clean.shape)

    for nfft in (512, 1024, 2048):
        out = apply_iterative_wiener(
            noisy, frame_len=nfft, lpc_order=20, iterations=2, alpha=0.8, thresh=0.01
        )
        # ignore edges to avoid overlap-add warm-up/cool-down effects
        center = slice(nfft, len(noisy) - nfft)
        rms_in = np.sqrt(np.mean(noisy[center] ** 2))
        rms_out = np.sqrt(np.mean(out[center] ** 2))
        ratio = rms_out / rms_in

        # amplitude should be preserved (Wiener gain ~ 1 for a clean signal)
        assert 0.5 < ratio < 1.5, (nfft, ratio)
        # and must not exhibit the old 1 / sqrt(nfft) collapse
        assert ratio > 5.0 / np.sqrt(nfft), (nfft, ratio)
