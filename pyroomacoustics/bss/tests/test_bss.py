
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

def test_bss(algo='auxiva', L=256):

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
    mics_signals = np.sum(separate_recordings, axis=0)

    ## STFT analysis
    # shape == (n_chan, n_frames, n_freq)
    X = pra.transform.analysis(mics_signals.T, L, L, zp_front=L//2, zp_back=L//2)

    ## START BSS
    if algo == 'auxiva':
        # Run AuxIVA
        Y = pra.bss.auxiva(X, n_iter=30, proj_back=True)
        max_mse = 1e-5
    elif algo == 'ilrma':
        # Run ILRMA
        Y = pra.bss.ilrma(X, n_iter=30, n_components=30, proj_back=True)
        max_mse = 1e-5
    elif algo == 'sparseauxiva':
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

    print('%s with a %d frame length: Relative MSE (expected less than %.e)'
          % (algo, L, max_mse), mse / input_variance)
    assert (mse / input_variance) < max_mse

def test_bss_auxiva_256():
    test_bss(algo='auxiva', L=256)

def test_bss_auxiva_512():
    test_bss(algo='auxiva', L=512)

def test_bss_auxiva_1024():
    test_bss(algo='auxiva', L=1024)

def test_bss_auxiva_2048():
    test_bss(algo='auxiva', L=2048)

def test_bss_auxiva_4096():
    test_bss(algo='auxiva', L=4096)

def test_bss_ilrma_256():
    test_bss(algo='ilrma', L=256)

def test_bss_ilrma_512():
    test_bss(algo='ilrma', L=512)

def test_bss_ilrma_1024():
    test_bss(algo='ilrma', L=1024)

def test_bss_ilrma_2048():
    test_bss(algo='ilrma', L=2048)

def test_bss_ilrma_4096():
    test_bss(algo='ilrma', L=4096)

def test_bss_sparseauxiva_256():
    test_bss(algo='sparseauxiva', L=256)

def test_bss_sparseauxiva_512():
    test_bss(algo='sparseauxiva', L=512)

def test_bss_sparseauxiva_1024():
    test_bss(algo='sparseauxiva', L=1024)

def test_bss_sparseauxiva_2048():
    test_bss(algo='sparseauxiva', L=2048)

def test_bss_sparseauxiva_4096():
    test_bss(algo='sparseauxiva', L=4096)

if __name__ == '__main__':
    # Test auxiva with frame lengths [256, 512, 1024, 2048, 4096]
    test_bss_auxiva_256()
    test_bss_auxiva_512()
    test_bss_auxiva_1024()
    test_bss_auxiva_2048()
    test_bss_auxiva_4096()
    # Test ilrma with frame lengths [256, 512, 1024, 2048, 4096]
    test_bss_ilrma_256()
    test_bss_ilrma_512()
    test_bss_ilrma_1024()
    test_bss_ilrma_2048()
    test_bss_ilrma_4096()
    # Test sparse auxiva with frame lengths [256, 512, 1024, 2048, 4096]
    test_bss_sparseauxiva_256()
    test_bss_sparseauxiva_512()
    test_bss_sparseauxiva_1024()
    test_bss_sparseauxiva_2048()
    test_bss_sparseauxiva_4096()
