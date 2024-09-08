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
