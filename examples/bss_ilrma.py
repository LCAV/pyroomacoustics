'''
Blind Source Separation with Independent Low-Rank Matrix Analysis (ILRMA)
=======================================================

Demonstrate how to do blind source separation (BSS) using the Independent
Low-Rank Matrix Analysis technique. The method implemented is described
in the following publication.

   D. Kitamura, N. Ono, H. Sawada, H. Kameoka, and H. Saruwatari *Determined Blind Source Separation
    with Independent Low-Rank Matrix Analysis*, in Audio Source Separation, S. Makino, Ed. Springer, 2018, pp.  125-156.

It works in the STFT domain. The test files were extracted from the
`CMU ARCTIC <http://www.festvox.org/cmu_arctic/>`_ corpus.
'Signal Separation Campaing 2011
<http://sisec2011.wiki.irisa.fr/tiki-index165d.html?page=Professionally+produced+music+recordings>
<http://sisec2011.wiki.irisa.fr/tiki-indexbfd7.html?page=Underdetermined+speech+and+music+mixtures>

Running this script will do two things.

1. It will separate the sources.
2. Show a plot of the clean and separated spectrograms
3. Show a plot of the SDR and SIR as a function of the number of iterations.
4. Create a `play(ch)` function that can be used to play the `ch` source (if you are in ipython say).

This script requires the `mir_eval`, and `sounddevice` packages to run.
'''

import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
import librosa

from mir_eval.separation import bss_eval_images
import sounddevice as sd

def select_dataset(dataset_name):
    wav_files = {
        'cmu_arctic_corpus': [
        ['input_samples/cmu_arctic_corpus/cmu_arctic_us_axb_a0004.wav',
            'input_samples/cmu_arctic_corpus/cmu_arctic_us_axb_a0005.wav',
            'input_samples/cmu_arctic_corpus/cmu_arctic_us_axb_a0006.wav',],
        ['input_samples/cmu_arctic_corpus/cmu_arctic_us_aew_a0001.wav',
            'input_samples/cmu_arctic_corpus/cmu_arctic_us_aew_a0002.wav',
            'input_samples/cmu_arctic_corpus/cmu_arctic_us_aew_a0003.wav',]
        ],

        'SiSec2011_Music_NZ': [
        ['input_samples/SiSec2011_Music/Ultimate NZ Tour/dev2__ultimate_nz_tour__snip_43_61__guitar.wav',],
        ['input_samples/SiSec2011_Music/Ultimate NZ Tour/dev2__ultimate_nz_tour__snip_43_61__synth.wav',]
        ],

        'SiSec2011_Music_Bearlin' : [
        ['input_samples/SiSec2011_Music/Bearlin_Roads/dev1__bearlin-roads__snip_85_99__acoustic_guit_main.wav',],
        ['input_samples/SiSec2011_Music/Bearlin_Roads/dev1__bearlin-roads__snip_85_99__vocals.wav',]
        ],

        'SiSec2011_Speech': [
        ['input_samples/SiSec2011_Speech/dev1_female4_src_1.wav',],
        ['input_samples/SiSec2011_Speech/dev1_male4_src_1.wav',]
        ],
    }
    return wav_files.get(dataset_name, "Invalid dataset")

wav_files = select_dataset('SiSec2011_Music_NZ')

if __name__ == '__main__':
    # STFT frame length
    L = 2048

    # Room 4m by 6m
    room_dim = [8, 9]

    # source location
    source = np.array([1, 4.5])

    # create an anechoic room with sources and mics
    room = pra.ShoeBox(
        room_dim,
        fs=wavfile.read(wav_files[0][0])[0],        # match sampling frequency of the input samples
        max_order=15,
        absorption=0.35,
        sigma2_awgn=1e-8)

    # get signals
    # signals = [np.concatenate([librosa.load(f, sr=16000)[0]
    #    for f in source_files])
    #    for source_files in wav_files ]

    signals = [np.concatenate([wavfile.read(f)[1].astype(np.float32)
               for f in source_files])
               for source_files in wav_files]

    # From stereo to mono
    if signals[0].ndim > 1:
        signals[0] = signals[0][:, 0]/2. + signals[0][:, 1]/2.
        signals[1] = signals[1][:, 0]/2. + signals[1][:, 1]/2.

    delays = [1., 0.]
    locations = [[2.5, 3], [2.5, 6]]

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
    window = np.sqrt(pra.hann(L))

    from mir_eval.separation import bss_eval_images
    ref = np.moveaxis(separate_recordings, 1, 2)
    SDR, SIR = [], []
    def convergence_callback(Y):
        Y = np.transpose(Y, [1, 0, 2])
        global SDR, SIR
        from mir_eval.separation import bss_eval_images
        ref = np.moveaxis(separate_recordings, 1, 2)
        y = np.array([pra.istft(Y[:,:,ch], L, L // 2,
            transform=np.fft.irfft, win=window) for ch in range(Y.shape[2])])
        sdr, isr, sir, sar, perm = bss_eval_images(ref[:,:y.shape[1],0], y)
        SDR.append(sdr)
        SIR.append(sir)

    # START BSS
    ###########
    # The STFT needs front *and* back padding


    # shape == (n_chan, n_frames, n_freq)
    X = np.array([pra.stft(ch, L, L // 2, win=window, transform=np.fft.rfft) for ch in mics_signals])
    X = np.moveaxis(X, 0, 2)

    # Run ILRMA
    Y, W = pra.bss.ilrma(X, n_iter=30, n_components=2, proj_back=True,
            return_filters=True, callback=convergence_callback)

    # run iSTFT
    y = np.array([pra.istft(Y[:,:,ch], L, L // 2, transform=np.fft.irfft, win=window) for ch in range(Y.shape[2])])

    # Compare SIR
    #############
    sdr, isr, sir, sar, perm = bss_eval_images(ref[:,:y.shape[1],0], y)

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

    plt.figure()
    a = np.array(SDR)
    b = np.array(SIR)
    plt.plot(np.arange(a.shape[0]) * 10, a[:,0], label='SDR Source 0', c='r', marker='*')
    plt.plot(np.arange(a.shape[0]) * 10, a[:,1], label='SDR Source 1', c='r', marker='o')
    plt.plot(np.arange(b.shape[0]) * 10, b[:,0], label='SIR Source 0', c='b', marker='*')
    plt.plot(np.arange(b.shape[0]) * 10, b[:,1], label='SIR Source 1', c='b', marker='o')
    plt.legend()

    plt.show()

    def play(ch):
        sd.play(pra.normalize(y[ch]) * 0.75, samplerate=room.fs, blocking=True)

