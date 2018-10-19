
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

def test_trinicon():
    '''
    This test do not check the correctness of the output of Trinicon.
    Only that it runs without errors.
    '''

    # STFT frame length
    L = 1024

    # Room 8m by 9m
    room_dim = [8, 9]

    # source location
    source = np.array([1, 4.5])

    # create the room with sources and mics
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
            pra.MicrophoneArray(np.c_[[6.5, 4.39], [6.5, 4.51]], fs=room.fs)
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

    # run Trinicon
    y,w = pra.bss.trinicon(mics_signals, filter_length=L, return_filters=True)


if __name__ == '__main__':
    # STFT frame length
    L = 1024

    # Room 8m by 9m
    room_dim = [8, 9]

    # source location
    source = np.array([1, 4.5])

    # create the room with sources and mics
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
            pra.MicrophoneArray(np.c_[[6.5, 4.39], [6.5, 4.51]], fs=room.fs)
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

    # START BSS
    ###########

    y,w = pra.bss.trinicon(mics_signals, filter_length=L, return_filters=True)
    y,w = pra.bss.trinicon(mics_signals, w0=w, return_filters=True)
    y,w = pra.bss.trinicon(mics_signals, w0=w, return_filters=True)
    y,w = pra.bss.trinicon(mics_signals, w0=w, return_filters=True)
    y,w = pra.bss.trinicon(mics_signals, w0=w, return_filters=True)
    y,w = pra.bss.trinicon(mics_signals, w0=w, return_filters=True)
    y = pra.bss.trinicon(mics_signals, w0=w)

    # Compare SIR
    #############
    sdr, isr, sir, sar, perm = bss_eval_images(ref[:,:,0], y[:,L//2:ref.shape[1]+L//2])

    print('SDR:', sdr)
    print('SIR:', sir)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(2,2,1)
    plt.specgram(ref[0,:,0], NFFT=1024, Fs=room.fs)
    plt.title('Reference 1')

    plt.subplot(2,2,2)
    plt.specgram(ref[1,:,0], NFFT=1024, Fs=room.fs)
    plt.title('Reference 2')

    plt.subplot(2,2,3)
    plt.specgram(y[perm[0],:], NFFT=1024, Fs=room.fs)
    plt.title('Source 1')

    plt.subplot(2,2,4)
    plt.specgram(y[perm[1],:], NFFT=1024, Fs=room.fs)
    plt.title('Source 2')

    plt.figure()
    a = np.array(SDR)
    b = np.array(SIR)
    plt.plot(np.arange(a.shape[0]) * 10, a, label='SDR')
    plt.plot(np.arange(b.shape[0]) * 10, b, label='SIR')
    plt.legend()

    plt.show()

    import sounddevice as sd
    def play(ch):
        sd.play(pra.normalize(y[ch]) * 0.75, samplerate=room.fs, blocking=True)


