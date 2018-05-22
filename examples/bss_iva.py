'''
Blind Source Separation with Indpendent Vector Analysis
=======================================================

Demonstrate how to do blind source separation (BSS) using the indpendent vector
analysis technique. The method implemented is described in the following
publication.

    N. Ono, *Stable and fast update rules for independent vector analysis based
    on auxiliary function technique*, Proc. IEEE, WASPAA, 2011.

It works in the STFT domain. The test files were extracted from the
`CMU ARCTIC <http://www.festvox.org/cmu_arctic/>`_ corpus.

Running this script will do two things.

1. It will separate the sources.
2. Show a plot of the clean and separated spectrograms
3. Show a plot of the SDR and SIR as a function of the number of iterations.
4. Create a `play(ch)` function that can be used to play the `ch` source (if you are in ipython say).

This script requires the `mir_eval` to run, and `tkinter` and `sounddevice` packages for the GUI option.
'''

import numpy as np
from scipy.io import wavfile

from mir_eval.separation import bss_eval_images

# We concatenate a few samples to make them long enough
wav_files = [
        ['examples/input_samples/cmu_arctic_us_axb_a0004.wav',
            'examples/input_samples/cmu_arctic_us_axb_a0005.wav',
            'examples/input_samples/cmu_arctic_us_axb_a0006.wav',],
        ['examples/input_samples/cmu_arctic_us_aew_a0001.wav',
            'examples/input_samples/cmu_arctic_us_aew_a0002.wav',
            'examples/input_samples/cmu_arctic_us_aew_a0003.wav',]
        ]

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Demonstration of blind source separation using IVA.')
    parser.add_argument('-b', '--block', type=int, default=2048,
            help='STFT block size')
    parser.add_argument('--gui', action='store_true',
            help='Creates a small GUI for easy playback of the sound samples')
    parser.add_argument('--save', action='store_true',
            help='Saves the output of the separation to wav files')

    args = parser.parse_args()

    if args.gui:
        # avoids a bug with tkinter and matplotlib
        import matplotlib
        matplotlib.use('TkAgg')

    import pyroomacoustics as pra


    # STFT frame length
    L = args.block

    # Room 4m by 6m
    room_dim = [8, 9]

    # source location
    source = np.array([1, 4.5])

    # create an anechoic room with sources and mics
    room = pra.ShoeBox(
        room_dim,
        fs=16000,
        max_order=15,
        absorption=0.35,
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
            transform=np.fft.irfft, zp_front=L//2, zp_back=L//2) for ch in range(Y.shape[2])])
        sdr, isr, sir, sar, perm = bss_eval_images(ref[:,:y.shape[1]-L//2,0], y[:,L//2:ref.shape[1]+L//2])
        SDR.append(sdr)
        SIR.append(sir)

    # START BSS
    ###########
    # The STFT needs front *and* back padding

    # shape == (n_chan, n_frames, n_freq)
    X = np.array([pra.stft(ch, L, L, transform=np.fft.rfft, zp_front=L//2, zp_back=L//2) for ch in mics_signals])
    X = np.moveaxis(X, 0, 2)

    # Run AuxIVA
    Y = pra.bss.auxiva(X, n_iter=30, proj_back=True, callback=convergence_callback)

    # run iSTFT
    y = np.array([pra.istft(Y[:,:,ch], L, L, transform=np.fft.irfft, zp_front=L//2, zp_back=L//2) for ch in range(Y.shape[2])])

    # Compare SIR
    #############
    sdr, isr, sir, sar, perm = bss_eval_images(ref[:,:y.shape[1]-L//2,0], y[:,L//2:ref.shape[1]+L//2])

    print('SDR:', sdr)
    print('SIR:', sir)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(2,2,1)
    plt.specgram(ref[0,:,0], NFFT=1024, Fs=room.fs)
    plt.title('Source 0 (clean)')

    plt.subplot(2,2,2)
    plt.specgram(ref[1,:,0], NFFT=1024, Fs=room.fs)
    plt.title('Source 1 (clean)')

    plt.subplot(2,2,3)
    plt.specgram(y[perm[0],:], NFFT=1024, Fs=room.fs)
    plt.title('Source 0 (separated)')

    plt.subplot(2,2,4)
    plt.specgram(y[perm[1],:], NFFT=1024, Fs=room.fs)
    plt.title('Source 1 (separated)')

    plt.tight_layout(pad=0.5)

    plt.figure()
    a = np.array(SDR)
    b = np.array(SIR)
    plt.plot(np.arange(a.shape[0]) * 10, a[:,0], label='SDR Source 0', c='r', marker='*')
    plt.plot(np.arange(a.shape[0]) * 10, a[:,1], label='SDR Source 1', c='r', marker='o')
    plt.plot(np.arange(b.shape[0]) * 10, b[:,0], label='SIR Source 0', c='b', marker='*')
    plt.plot(np.arange(b.shape[0]) * 10, b[:,1], label='SIR Source 1', c='b', marker='o')
    plt.legend()

    plt.tight_layout(pad=0.5)

    if not args.gui:
        plt.show()
    else:
        plt.show(block=False)

    if args.save:
        from scipy.io import wavfile

        wavfile.write('bss_iva_mix.wav', room.fs,
                pra.normalize(mics_signals[0,:], bits=16).astype(np.int16))
        for i, sig in enumerate(y):
            wavfile.write('bss_iva_source{}.wav'.format(i+1), room.fs,
                    pra.normalize(sig, bits=16).astype(np.int16))

    if args.gui:

        # Make a simple GUI to listen to the separated samples
        from tkinter import Tk, Button, Label
        import sounddevice as sd

        # Now come the GUI part
        class PlaySoundGUI(object):
            def __init__(self, master, fs, mix, sources):
                self.master = master
                self.fs = fs
                self.mix = mix
                self.sources = sources
                master.title("A simple GUI")

                self.label = Label(master, text="This is our first GUI!")
                self.label.pack()

                self.mix_button = Button(master, text='Mix', command=lambda: self.play(self.mix))
                self.mix_button.pack()

                self.buttons = []
                for i, source in enumerate(self.sources):
                    self.buttons.append(Button(master, text='Source ' + str(i+1), command=lambda src=source : self.play(src)))
                    self.buttons[-1].pack()

                self.stop_button = Button(master, text="Stop", command=sd.stop)
                self.stop_button.pack()

                self.close_button = Button(master, text="Close", command=master.quit)
                self.close_button.pack()

            def play(self, src):
                sd.play(pra.normalize(src) * 0.75, samplerate=self.fs, blocking=False)


        root = Tk()
        my_gui = PlaySoundGUI(root, room.fs, mics_signals[0,:], y)
        root.mainloop()
