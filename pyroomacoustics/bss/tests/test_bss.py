
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile

# We use several sound samples for each source to have a long enough length
wav_files = [
        [
            'examples/input_samples/cmu_arctic_us_axb_a0004.wav',
            'examples/input_samples/cmu_arctic_us_axb_a0005.wav',
            'examples/input_samples/cmu_arctic_us_axb_a0006.wav',
            ],
        [
            'examples/input_samples/cmu_arctic_us_aew_a0001.wav',
            'examples/input_samples/cmu_arctic_us_aew_a0002.wav',
            'examples/input_samples/cmu_arctic_us_aew_a0003.wav',
            ],
        ]
# If True test the bss algorithm
test_auxiva = False
test_ilrma = True
test_sparseauxiva = True
choices = ['auxIVA', 'ILRMA', 'sparseauxIVA']

# List of frame lengths to test
L = [256, 512, 1024, 2048]


def test_bss(algo,L):

    # Room dimensions in meters
    room_dim = [8, 9]

    # create a room with sources and mics
    room = pra.ShoeBox(
        room_dim,
        fs=16000,
        max_order=0,
        sigma2_awgn=1e-8)

    # get signals
    signals = [ np.concatenate([wavfile.read(f)[1].astype(np.float32)
        for f in source_files])
        for source_files in wav_files ]
    delays = [1., 0.]
    locations = [[2.5,3], [2.5, 6]]

    # add mic and good source to room
    # Add silent signals to all sources
    for sig, d, loc in zip(signals, delays, locations):
        room.add_source(loc, signal=np.zeros_like(sig), delay=d)

    # add microphone array
    room.add_microphone_array(
            pra.MicrophoneArray(np.c_[[6.5, 4.49], [6.5, 4.51]], fs=room.fs)
            )

    # compute RIRs
    room.compute_rir()

    # Record each source separately
    separate_recordings = []
    for source, signal in zip(room.sources, signals):

        source.signal[:] = signal

        room.simulate()
        separate_recordings.append(room.mic_array.signals)

        source.signal[:] = 0.
    separate_recordings = np.array(separate_recordings)

    # Mix down the recorded signals
    limit = separate_recordings.shape[2] - (separate_recordings.shape[2] % L)
    mics_signals = np.sum(separate_recordings[:,:,:limit], axis=0)

    ## STFT analysis
    # shape == (n_chan, n_frames, n_freq)
    X = pra.transform.analysis(mics_signals.T, L, L, zp_front=L//2, zp_back=L//2)

    X_test = np.array([pra.stft(ch, L, L, transform=np.fft.rfft, zp_front=L // 2, zp_back=L // 2) for ch in mics_signals])
    X_test = np.moveaxis(X_test, 0, 2)
    ## START BSS
    if choices[algo] == 'auxIVA':
        # Run AuxIVA
        Y = pra.bss.auxiva(X, n_iter=30, proj_back=True)
        max_mse = 1e-5
    elif choices[algo] == 'ILRMA':
        # Run ILRMA
        Y = pra.bss.ilrma(X, n_iter=30, n_components=30, proj_back=True)
        max_mse = 1e-5
    elif choices[algo] == 'sparseauxIVA':
        # Estimate set of active frequency bins
        ratio = 0.35
        average = np.abs(np.mean(np.mean(X, axis=2), axis=0))
        k = np.int_(average.shape[0] * ratio)
        S = np.sort(np.argpartition(average, -k)[-k:])
        # Run SparseAuxIva
        Y = pra.bss.sparseauxiva(X, S, n_iter=30, proj_back=True)
        max_mse = 1e-4

    ## STFT Synthesis
    y = pra.transform.synthesis(Y, L, L, zp_front=L//2, zp_back=L//2).T

    # Calculate MES
    #############
    ref = np.moveaxis(separate_recordings, 1, 2)
    y_aligned = y[:,L//2:ref.shape[1]+L//2]

    mse = np.mean((ref[:,:y_aligned.shape[1],0] - y_aligned)**2)
    input_variance = np.var(np.concatenate(signals))

    print('%s with frame length of %d: Relative MSE (expect less than %f)'
          % (choices[algo], L, max_mse), mse / input_variance)
    assert (mse / input_variance) < max_mse

if __name__ == '__main__':
    if test_auxiva:
        # Test auxIVA
        for block in L:
            test_bss(algo=0,L=block)
    if test_ilrma:
        # Test ILRMA
        for block in L:
            test_bss(algo=1,L=block)
    if test_sparseauxiva:
        # Test Sparse auxIVA
        for block in L:
            test_bss(algo=2,L=block)
