#unittest import TestCase

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy.io import wavfile
from scipy.signal import resample,fftconvolve

import pyroomacoustics as pra

class TestTimeDomain(TestCase):
        
    def test_good_result(self):
        # Spectrogram figure properties
        figsize=(15, 7)        # figure size
        fft_size = 512         # fft size for analysis
        fft_hop  = 8           # hop between analysis frame
        fft_zp = 512           # zero padding
        analysis_window = np.concatenate((pra.hann(fft_size), np.zeros(fft_zp)))
        t_cut = 0.83           # length in [s] to remove at end of signal (no sound)

        # Some simulation parameters
        Fs = 8000
        t0 = 1./(Fs*np.pi*1e-2)  # starting time function of sinc decay in RIR response
        absorption = 0.90
        max_order_sim = 10
        sigma2_n = 5e-7

        # Room 1 : Shoe box
        room_dim = [4, 6]

        # the good source is fixed for all 
        good_source = np.array([1, 4.5])           # good source
        normal_interferer = np.array([2.8, 4.3])   # interferer
        hard_interferer = np.array([1.5, 3])       # interferer in direct path
        #normal_interferer = hard_interferer

        # microphone array design parameters
        mic1 = np.array([2, 1.5])   # position
        M = 8                       # number of microphones
        d = 0.08                    # distance between microphones
        phi = 0.                    # angle from horizontal
        max_order_design = 1        # maximum image generation used in design
        shape = 'Linear'            # array shape
        Lg_t = 0.100                # Filter size in seconds
        Lg = np.ceil(Lg_t*Fs)       # Filter size in samples
        delay = 0.050               # Beamformer delay in seconds

        # define the FFT length
        N = 1024

        # create a microphone array
        if shape is 'Circular':
            R = pra.circular2DArray(mic1, M, phi, d*M/(2*np.pi)) 
        else:
            R = pra.linear2DArray(mic1, M, phi, d) 

        # The first signal (of interest) is singing
        rate1, signal1 = wavfile.read('samples/singing_'+str(Fs)+'.wav')
        signal1 = np.array(signal1, dtype=float)
        signal1 = pra.normalize(signal1)
        signal1 = pra.highpass(signal1, Fs)
        delay1 = 0.

        # the second signal (interferer) is some german speech
        rate2, signal2 = wavfile.read('samples/german_speech_'+str(Fs)+'.wav')
        signal2 = np.array(signal2, dtype=float)
        signal2 = pra.normalize(signal2)
        signal2 = pra.highpass(signal2, Fs)
        delay2 = 1.

        # create the room with sources and mics
        room1 = pra.Room.shoeBox2D(
            [0,0],
            room_dim,
            Fs,
            t0 = t0,
            max_order=max_order_sim,
            absorption=absorption,
            sigma2_awgn=sigma2_n)

        # add mic and good source to room
        room1.addSource(good_source, signal=signal1, delay=delay1)
        room1.addSource(normal_interferer, signal=signal2, delay=delay2)

        # obtain the desired order sources
        good_sources = room1.sources[0][:max_order_design+1]
        bad_sources = room1.sources[1][:max_order_design+1]

        '''
        MVDR direct path only simulation
        '''

        # compute beamforming filters
        mics = pra.Beamformer(R, Fs, N, Lg=Lg)
        room1.addMicrophoneArray(mics)
        room1.compute_RIR()
        room1.simulate()
        mics.rakeMVDRFilters(room1.sources[0][0:1],
                room1.sources[1][0:1],
                            sigma2_n*np.eye(mics.Lg*mics.M), delay=delay)

        # process the signal
        output = mics.process()

        # save to output file
        input_mic = pra.normalize(pra.highpass(mics.signals[mics.M/2], Fs))
        wavfile.write('output_samples/input.wav', Fs, input_mic)

        out_DirectMVDR = pra.normalize(pra.highpass(output, Fs))
        wavfile.write('output_samples/output_DirectMVDR.wav', Fs, out_DirectMVDR)


        '''
        Rake MVDR simulation
        '''

        # compute beamforming filters
        mics = pra.Beamformer(R, Fs, N, Lg=Lg)
        room1.addMicrophoneArray(mics)
        room1.compute_RIR()
        room1.simulate()
        mics.rakeMVDRFilters(good_sources, 
                            bad_sources, 
                            sigma2_n*np.eye(mics.Lg*mics.M), delay=delay)

        # process the signal
        output = mics.process()

        # save to output file
        out_RakeMVDR = pra.normalize(pra.highpass(output, Fs))
        wavfile.write('output_samples/output_RakeMVDR.wav', Fs, out_RakeMVDR)

        '''
        Perceptual direct path only simulation
        '''

        # compute beamforming filters
        mics = pra.Beamformer(R, Fs, N, Lg=Lg)
        room1.addMicrophoneArray(mics)
        room1.compute_RIR()
        room1.simulate()
        mics.rakePerceptualFilters(room1.sources[0][0:1],
                room1.sources[1][0:1],
                            sigma2_n*np.eye(mics.Lg*mics.M), delay=delay)

        # process the signal
        output = mics.process()

        # save to output file
        out_DirectPerceptual = pra.normalize(pra.highpass(output, Fs))
        wavfile.write('output_samples/output_DirectPerceptual.wav', Fs, out_DirectPerceptual)

        '''
        Rake Perceptual simulation
        '''

        # compute beamforming filters
        mics = pra.Beamformer(R, Fs, N, Lg=Lg)
        room1.addMicrophoneArray(mics)
        room1.compute_RIR()
        room1.simulate()
        mics.rakePerceptualFilters(good_sources, 
                            bad_sources, 
                            sigma2_n*np.eye(mics.Lg*mics.M), delay=delay)

        # process the signal
        output = mics.process()

        # save to output file
        out_RakePerceptual = pra.normalize(pra.highpass(output, Fs))
        wavfile.write('output_samples/output_RakePerceptual.wav', Fs, out_RakePerceptual)

        '''
        Plot all the spectrogram
        '''

        dSNR = pra.dB(room1.dSNR(mics.center[:,0], source=0), power=True)
        self.assertTrue(np.allclose(13.7113358268, dSNR))