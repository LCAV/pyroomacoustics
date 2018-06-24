
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile

wav_files = [ 
    'examples/input_samples/german_speech_8000.wav',
    'examples/input_samples/singing_8000.wav',
    ]

#def test_auxiva():

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
        fs=8000,
        max_order=0,
        absorption=0.45,
        sigma2_awgn=1e-8)

    # get signals
    signals = [wavfile.read(f)[1].astype(np.float32) for f in wav_files]
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

    # Monitor Convergence
    #####################

    from mir_eval.separation import bss_eval_images
    ref = np.moveaxis(separate_recordings, 1, 2)
    SDR, SIR = [], []
    def convergence_callback(Y):
        global SDR, SIR
        from mir_eval.separation import bss_eval_images
        ref = np.moveaxis(separate_recordings, 1, 2)
        y = np.array([pra.istft(Y[:,:,ch], L, L, 
            transform=np.fft.irfft, zp_back=L) for ch in range(Y.shape[2])])
        sdr, isr, sir, sar, perm = bss_eval_images(ref[:,:,0], y[:,:ref.shape[1]])
        SDR.append(sdr)
        SIR.append(sir)

    # START BSS
    ###########

    y,w = pra.bss.trinicon(mics_signals, filter_length=L, return_filters=True)
    y,w = pra.bss.trinicon(mics_signals, w0=w, return_filters=True)
    y,w = pra.bss.trinicon(mics_signals, w0=w, return_filters=True)
    y,w = pra.bss.trinicon(mics_signals, w0=w, return_filters=True)
    y,w = pra.bss.trinicon(mics_signals, w0=w, return_filters=True)
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


