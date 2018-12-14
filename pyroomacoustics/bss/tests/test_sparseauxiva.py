# -*- coding: utf-8 -*-
import numpy as np
from scipy.io import wavfile
import pyroomacoustics as pra
from mir_eval.separation import bss_eval_images

# Blind Source Separation techniques such as Independent Vector Analysis (IVA)
# using an Auxiliary function are implemented in ´pyroomacoustics´. IVA based
# algorithms work when the number of microphones is the same as the number of
# sources, i.e., the determinant case. Through this example, we will deal with
# the case of 2 sources and 2 microphones.

# First, open and concatanate wav files from the CMU dataset.
# concatanate audio samples to make them look long enough
wav_files = [
    ['../../../examples/input_samples/cmu_arctic_us_axb_a0004.wav',
     '../../../examples/input_samples/cmu_arctic_us_axb_a0005.wav',
     '../../../examples/input_samples/cmu_arctic_us_axb_a0006.wav', ],
    ['../../../examples/input_samples/cmu_arctic_us_aew_a0001.wav',
     '../../../examples/input_samples/cmu_arctic_us_aew_a0002.wav',
     '../../../examples/input_samples/cmu_arctic_us_aew_a0003.wav', ]
]


def test_sparseauxiva():
    fs = 16000

    signals = [np.concatenate([wavfile.read(f)[1].astype(np.float32, order='C') for f in source_files])
               for source_files in wav_files]

    wavfile.write('sample1.wav', fs, np.asarray(signals[0], dtype=np.int16))
    wavfile.write('sample2.wav', fs, np.asarray(signals[1], dtype=np.int16))

    # Define an anechoic room envrionment, as well as the microphone array and source locations.

    # Room 4m by 6m
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

    # save mixed signals as wav files
    wavfile.write('mix1.wav', fs, np.asarray(mics_signals[0].T, dtype=np.int16))
    wavfile.write('mix2.wav', fs, np.asarray(mics_signals[1].T, dtype=np.int16))
    wavfile.write('mix1_norm.wav', fs,
                  np.asarray(mics_signals[0].T / np.max(np.abs(mics_signals[0].T)) * 32767, dtype=np.int16))
    wavfile.write('mix2_norm.wav', fs,
                  np.asarray(mics_signals[1].T / np.max(np.abs(mics_signals[1].T)) * 32767, dtype=np.int16))

    # STFT frame length
    L = 2048

    # START BSS
    ###########

    # Observation vector in the STFT domain
    X = np.array([pra.stft(ch, L, L, transform=np.fft.rfft, zp_front=L // 2, zp_back=L // 2) for ch in mics_signals])
    X = np.moveaxis(X, 0, 2)

    # Reference signal to calculate performance of BSS

    ref = np.moveaxis(separate_recordings, 1, 2)

    ratio = 0.35
    average = np.abs(np.mean(np.mean(X, axis=2), axis=0))
    k = np.int_(average.shape[0] * ratio)
    S = np.argpartition(average, -k)[-k:]
    S = np.sort(S)
    n_iter = 30

    # Run SparseAuxIva
    Y = pra.bss.sparseauxiva(X, S, n_iter, lasso=True)

    # run iSTFT
    y = np.array([pra.istft(Y[:, :, ch], L, L, transform=np.fft.irfft, zp_front=L // 2, zp_back=L // 2) for ch in
                  range(Y.shape[2])])

    # Compare SIR and SDR with our reference signal
    sdr, isr, sir, sar, perm = bss_eval_images(ref[:, :y.shape[1] - L // 2, 0], y[:, L // 2:ref.shape[1] + L // 2])
    print('SDR: {0}, SIR: {1}'.format(sdr, sir))

    wavfile.write('demix1.wav', fs, np.asarray(y[0].T, dtype=np.int16))
    wavfile.write('demix2.wav', fs, np.asarray(y[1].T, dtype=np.int16))
    wavfile.write('demix1_norm.wav', fs, np.asarray(y[0].T / np.max(np.abs(y[0].T)) * 32767, dtype=np.int16))
    wavfile.write('demix2_norm.wav', fs, np.asarray(y[1].T / np.max(np.abs(y[1].T)) * 32767, dtype=np.int16))


if __name__ == '__main__':
    test_sparseauxiva()
