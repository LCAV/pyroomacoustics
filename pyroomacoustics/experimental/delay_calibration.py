from __future__ import division, print_function

import numpy as np
from scipy import signal

from .physics import calculate_speed_of_sound

class DelayCalibration:

    def __init__(self, fs, pad_time=0., mls_bits=16, repeat=1, 
                 temperature=25., humidity=50., pressure=1000.):
        '''
        Initialize the delay calibration object

        Parameters
        ----------
        fs : int
            Sampling frequency to use
        pad_time : float
            Duration of silence to add at the end of the test signal (i.e. expected T60)
        mls_bits : int
            Number of bits to use for the maximum length sequence
        repeat : int
            Number of repetition of the measurement before averaging
        temperature : float
            Room temperature
        humidity : float
            Ambient humidity
        pressure : float
            Atmospheric pressure
        '''
            

        self.fs = fs
        self.mls_bits = mls_bits
        self.repeat = repeat
        self.pad_time = pad_time

        self.temperature = temperature
        self.humidity = humidity
        self.pressure = pressure
        self.c = calculate_speed_of_sound(temperature, humidity, pressure)

    def run(self, distance=0., ch_in=None, ch_out=None, oversampling=1):
        '''
        Run the calibration. Plays a maximum length sequence and cross correlate
        the signals to find the time delay.

        Parameters
        ----------
        distance : float, optional
            Distance between the speaker and microphone
        ch_in : int, optional
            The input channel to use. If not specified, all channels are calibrated
        ch_out : int, optional
            The output channel to use. If not specified, all channels are calibrated
        '''

        if ch_out is None:
            ch_out = [0]

        # create the maximum length sequence
        mls = 0.95*np.array(2*signal.max_len_seq(self.mls_bits)[0] - 1, dtype=np.float32)

        # output signal
        s = np.zeros((mls.shape[0] + int(self.pad_time*self.fs), sd.default.channels[1]), dtype=np.float32)

        # placeholder for delays
        delays = np.zeros((sd.default.channels[0], len(ch_out)))

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            import warnings
            warnings.warn('Matplotlib is required for plotting')
            return

        for och in ch_out:
            # play and record the signal simultaneously
            s[:mls.shape[0], och] = 0.1 * mls
            rec_sig = sd.playrec(s, self.fs, channels=sd.default.channels[0], blocking=True)
            s[:,och] = 0

            for ich in range(rec_sig.shape[1]):
                xcorr = signal.correlate(rec_sig[:,ich], mls, mode='valid')
                delays[ich,och] = np.argmax(xcorr)
                plt.plot(xcorr)
                plt.show()

        # subtract distance
        delays -= int(distance / self.c * self.fs)

        return delays


if __name__ == "__main__":

    try:
        import sounddevice as sd

        sd.default.device = (2,2)
        sd.default.channels = (1,2)
        dc = DelayCalibration(48000, mls_bits=16, pad_time=0.5, repeat=1, temperature=25.6, humidity=30.)

        delays = dc.run(ch_out=[0,1])

        print(delays)
    except:
        raise ImportError('Sounddevice package must be installed to run that script.')

