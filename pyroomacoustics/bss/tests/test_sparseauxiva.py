
import numpy as np
from scipy.io import wavfile
import pyroomacoustics as pra

# We use several sound samples for each source to have a length long enough

wav_files = [
    ['examples/input_samples/cmu_arctic_us_aew_a0001.wav',
     'examples/input_samples/cmu_arctic_us_aew_a0002.wav',
     'examples/input_samples/cmu_arctic_us_aew_a0003.wav',],
    ['examples/input_samples/cmu_arctic_us_axb_a0004.wav',
     'examples/input_samples/cmu_arctic_us_axb_a0005.wav',
     'examples/input_samples/cmu_arctic_us_axb_a0006.wav',]
    ]


def test_sparseauxiva():

    signals = [np.concatenate([wavfile.read(f)[1].astype(np.float32, order='C')
               for f in source_files])
               for source_files in wav_files]

    # Define an anechoic room envrionment, as well as the microphone array and source locations.
    # Room dimensions in meters
    room_dim = [8, 9]
    # source locations and delays
    locations = [[2.5, 3], [2.5, 6]]
    delays = [1., 0.]
    # create an anechoic room with sources and mics
    room = pra.ShoeBox(room_dim, fs=16000, max_order=15, absorption=0.35, sigma2_awgn=1e-8)

    # add mic and good source to room
    # Add silent signals to all sources
    for sig, d, loc in zip(signals, delays, locations):
        room.add_source(loc, signal=np.zeros_like(sig), delay=d)

    # add microphone array
    room.add_microphone_array(pra.MicrophoneArray(np.c_[[6.5, 4.49], [6.5, 4.51]], room.fs))

    # Compute the RIRs as in the Room Impulse Response generation section.

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

    # STFT frame length
    L = 2048

    # Observation vector in the STFT domain
    X = np.array([pra.stft(ch, L, L, transform=np.fft.rfft, zp_front=L // 2, zp_back=L // 2)
                  for ch in mics_signals])
    X = np.moveaxis(X, 0, 2)

    # START BSS
    ###########
    # Estimate set of active frequency bins
    ratio = 0.35
    average = np.abs(np.mean(np.mean(X, axis=2), axis=0))
    k = np.int_(average.shape[0] * ratio)
    S = np.argpartition(average, -k)[-k:]
    S = np.sort(S)

    # Run SparseAuxIva
    Y = pra.bss.sparseauxiva(X, S)
    ###########

    # run iSTFT
    y = np.array([pra.istft(Y[:, :, ch], L, L, transform=np.fft.irfft, zp_front=L // 2, zp_back=L // 2)
                  for ch in range(Y.shape[2])])

    # Compare SIR
    #############
    ref = np.moveaxis(separate_recordings, 1, 2)
    y_aligned = y[:,L//2:ref.shape[1]+L//2]

    mse = np.mean((ref[:,:,0] - y_aligned)**2)
    input_variance = np.var(np.concatenate(signals))

    print('Relative MSE (expect less than 1e-3):', mse / input_variance)

    assert (mse / input_variance) < 1e-3


if __name__ == '__main__':
    test_sparseauxiva()