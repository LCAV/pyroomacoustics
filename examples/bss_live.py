'''
Live Demonstration of Blind Source Separation
=============================================

Demonstrate the performance of different blind source separation (BSS) algorithms:

1) Independent Vector Analysis (IVA)
The method implemented is described in the following publication.

    N. Ono, *Stable and fast update rules for independent vector analysis based
    on auxiliary function technique*, Proc. IEEE, WASPAA, 2011.

2) Independent Low-Rank Matrix Analysis (ILRMA)
The method implemented is described in the following publications

    D. Kitamura, N. Ono, H. Sawada, H. Kameoka, H. Saruwatari, *Determined blind
    source separation unifying independent vector analysis and nonnegative matrix
    factorization,* IEEE/ACM Trans. ASLP, vol. 24, no. 9, pp. 1626-1641, September 2016

    D. Kitamura, N. Ono, H. Sawada, H. Kameoka, and H. Saruwatari *Determined Blind
    Source Separation with Independent Low-Rank Matrix Analysis*, in Audio Source Separation,
    S. Makino, Ed. Springer, 2018, pp.  125-156.

3) Sparse Independent Vector Analysis based on auxiliary function (SparseAuxIVA)
The method implemented is described in the following publication

    J. Jansky, Z. Koldovsky, and N. Ono *A computationally cheaper method for blind speech
    separation based on AuxIVA and incomplete demixing transform*, Proc. IEEE, IWAENC, 2016.

4) Fast Multichannel Nonnegative Matrix Factorization (FastMNMF)
The method implemented is described in the following publication

    K. Sekiguchi, A. A. Nugraha, Y. Bando, K. Yoshii, *Fast Multichannel Source 
    Separation Based on Jointly Diagonalizable Spatial Covariance Matrices*, EUSIPCO, 2019.

All the algorithms work in the STFT domain. The test files are recorded by an external
microphone array.
Depending on the input arguments running this script will do these actions:.

1. Record source signals from a connected microphone array
2. Separate the sources.
3. Create a `play(ch)` function that can be used to play the `ch` source (if you are in ipython say).
4. Save the separated sources as .wav files
5. Show a GUI where a mixed signals and the separated sources can be played

This script requires the `sounddevice` packages to run.
'''

import numpy as np

# important to avoid a crash when tkinter is called
import matplotlib
matplotlib.use('TkAgg')

import pyroomacoustics as pra

from tkinter import Tk, Label, Button
import sounddevice as sd

if __name__ == '__main__':

    choices = ['ilrma', 'auxiva', 'sparseauxiva', 'fastmnmf']

    import argparse
    parser = argparse.ArgumentParser(description='Records a segment of speech and then performs separation')
    parser.add_argument('-b', '--block', type=int, default=2048,
            help='STFT block size')
    parser.add_argument('-a', '--algo', type=str, default=choices[0], choices=choices,
            help='Chooses BSS method to run')
    parser.add_argument('-D', '--device', type=int,
            help='The sounddevice recording device id (obtain it with `python -m sounddevice`)')
    parser.add_argument('-d', '--duration', type=float,
            help='Recording time in seconds')
    parser.add_argument('-i', '--n_iter', type=int, default=20,
            help='Number of iteration of the algorithm')
    args = parser.parse_args()

    ## Prepare one-shot STFT
    L = args.block
    # Let's hard code sampling frequency to avoid some problems
    fs = 16000

    ## RECORD
    if args.device is not None:
        sd.default.device[0] = args.device

    ## MIXING
    print('* Recording started... ', end='')
    mics_signals = sd.rec(int(args.duration * fs), samplerate=fs, channels=2, blocking=True)
    print('done')

    ## STFT ANALYSIS
    # shape == (n_chan, n_frames, n_freq)
    X = pra.transform.analysis(mics_signals.T, L, L, zp_back=L//2, zp_front=L//2)

    ## Monitor convergence
    it = 10
    def cb_print(*args):
        global it
        print('  AuxIVA Iter', it)
        it += 10

    ## Run live BSS
    print('* Starting BSS')
    bss_type = args.algo
    if bss_type == 'auxiva':
        # Run AuxIVA
        Y = pra.bss.auxiva(X, n_iter=args.n_iter, proj_back=True, callback=cb_print)
    elif bss_type == 'ilrma':
        # Run ILRMA
        Y = pra.bss.ilrma(X, n_iter=args.n_iter, n_components=30, proj_back=True,
            callback=cb_print)
    elif bss_type == 'fastmnmf':
        # Run FastMNMF
        Y = pra.bss.fastmnmf(X, n_iter=args.n_iter, n_components=8, n_src=2,
            callback=cb_print)
    elif bss_type == 'sparseauxiva':
        # Estimate set of active frequency bins
        ratio = 0.35
        average = np.abs(np.mean(np.mean(X, axis=2), axis=0))
        k = np.int_(average.shape[0] * ratio)
        S = np.sort(np.argpartition(average, -k)[-k:])
        # Run SparseAuxIva
        Y = pra.bss.sparseauxiva(X, S, n_iter=30, proj_back=True,
                             callback=cb_print)

    ## STFT Synthesis
    y = pra.transform.synthesis(Y, L, L, zp_back=L//2, zp_front=L//2).T

    ## GUI starts
    print('* Start GUI')
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
    my_gui = PlaySoundGUI(root, fs, mics_signals[:,0], y)
    root.mainloop()
