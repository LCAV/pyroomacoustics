
import numpy as np 
try:
    import sounddevice as sd
    sounddevice_available = True
except ImportError:
    sounddevice_available = False

from .signals import exponential_sweep, linear_sweep
from .deconvolution import wiener_deconvolve

_sweep_types = {
        'exponential': exponential_sweep,
        'linear': linear_sweep,
        }

def measure_ir(sweep_length=1., sweep_type='exponential',
        fs=48000, f_lo=0., f_hi=None, 
        volume=0.9, pre_delay=0., post_delay=0.1, fade_in_out=0.,
        dev_in=None, dev_out=None, channels_input_mapping=None, channels_output_mapping=None,
        ascending=False, deconvolution=True, plot=True):
    '''
    Measures an impulse response by playing a sweep and recording it using the sounddevice package.

    Parameters
    ----------
    sweep_length: float, optional
        length of the sweep in seconds
    sweep_type: SweepType, optional
        type of sweep to use linear or exponential (default)
    fs: int, optional
        sampling frequency (default 48 kHz)
    f_lo: float, optional
        lowest frequency in the sweep
    f_hi: float, optional
        highest frequency in the sweep, can be a negative offset from fs/2
    volume: float, optional
        multiply the sweep by this number before playing (default 0.9)
    pre_delay: float, optional
        delay in second before playing sweep
    post_delay: float, optional
        delay in second before stopping recording after playing the sweep
    fade_in_out: float, optional
        length in seconds of the fade in and out of the sweep (default 0.)
    dev_in: int, optional
        input device number
    dev_out: int, optional
        output device number
    channels_input_mapping: array_like, optional
        List of channel numbers (starting with 1) to record. If mapping is
        given, channels is silently ignored.
    channels_output_mapping: array_like, optional
        List of channel numbers (starting with 1) where the columns of data
        shall be played back on. Must have the same length as number of
        channels in data (except if data is mono, in which case the signal is
        played back on all given output channels). Each channel number may only
        appear once in mapping.
    ascending: bool, optional
        wether the sweep is from high to low (default) or low to high frequencies
    deconvolution: bool, optional
        if True, apply deconvolution to the recorded signal to remove the sweep (default 0.)
    plot: bool, optional
        plot the resulting signal

    Returns
    -------
        Returns the impulse response if `deconvolution == True` and the recorded signal if not
    '''

    if not sounddevice_available:
        raise ImportError('Sounddevice package not availble. Install it to use this function.')

    N = int(sweep_length * fs) + 1

    sweep_func = _sweep_types[sweep_type]

    sweep = sweep_func(sweep_length, fs, f_lo=f_lo, f_hi=f_hi, fade=fade_in_out, ascending=ascending)

    # adjust the amplitude
    sweep *= volume

    # zero pad as needed
    pre_zeros = int(pre_delay * fs)
    post_zeros = int(post_delay * fs)
    test_signal = np.concatenate(
            (np.zeros(pre_zeros), sweep, np.zeros(post_zeros)) )

    # setup audio interface parameters
    if channels_input_mapping is None:
        channels_input_mapping = [1]
    if channels_output_mapping is None:
        channels_output_mapping = [1]
    if dev_in is not None:
        sd.default.device[0] = dev_in
    if dev_out is not None:
        sd.default.device[1] = dev_out

    # repeat if we need to play in multiple channels
    if len(channels_output_mapping) > 1:
        play_signal = np.tile(test_signal[:, np.newaxis], 
                              (1, len(channels_output_mapping)))
    else:
        play_signal = test_signal

    recorded_signal = sd.playrec(
            test_signal, samplerate=fs, 
            input_mapping=channels_input_mapping,
            output_mapping=channels_output_mapping,
            blocking=True
            )

    h = None
    if deconvolution:
        h = np.array(
                [wiener_deconvolve(recorded_signal[:,c], sweep) 
                for c in range(len(channels_input_mapping))]
                ).T

    if plot:
        import matplotlib.pyplot as plt

        if h is not None:
            plt.figure()
            plt.subplot(1,2,1)
            plt.plot(np.arange(h.shape[0]) / fs, h)
            plt.title('Impulse Response')
            plt.subplot(1,2,2)
            freq = np.arange(h.shape[0] // 2 + 1) * fs / h.shape[0]
            plt.plot(freq, 20.*np.log10(np.abs(np.fft.rfft(h, axis=0))))
            plt.title('Frequency content')
            
        plt.figure()
        plt.subplot(1,2,1)
        plt.plot(np.arange(recorded_signal.shape[0]) / fs, recorded_signal)
        plt.title('Recorded signal')
        plt.subplot(1,2,2)
        freq = np.arange(recorded_signal.shape[0] // 2 + 1) * fs / recorded_signal.shape[0]
        plt.plot(freq, 20.*np.log10(np.abs(np.fft.rfft(recorded_signal, axis=0))))
        plt.title('Frequency content')

        plt.show()

    if deconvolution:
        return recorded_signal, h
    else:
        return recorded_signal


if __name__ == '__main__':

    '''
    This is just an interface to measure impulse responses easily
    '''

    import argparse

    parser = argparse.ArgumentParser(prog='measure_ir', description='Measures an impulse response by playing a sweep and recording it using the sounddevice package.')
    parser.add_argument('-l', '--length', type=float, default=1.,
            help='length of the sweep in seconds')
    parser.add_argument('-t', '--type', type=str, default='exponential', 
            choices=_sweep_types.keys(),
            help='type of sweep to use linear or exponential (default)')
    parser.add_argument('-f', '--file', type=str,
            help='name of file where to save the recorded signals (without extension)')
    parser.add_argument('-r', '--fs', type=int, default=48000,
            help='sampling frequency (default 48 kHz)')
    parser.add_argument('--f-lo', type=float, default=0.,
            help='lowest frequency in the sweep')
    parser.add_argument('--f-hi', type=float, default=None,
            help='highest frequency in the sweep, can be a negative offset from fs/2')
    parser.add_argument('-v', '--volume', type=float, default=0.9,
            help='multiply the sweep by this number before playing (default 0.9)')
    parser.add_argument('--pre-delay', type=float, default=0.,
            help='delay in second before playing sweep')
    parser.add_argument('--post-delay', type=float, default=0.5,
            help='delay in second before stopping recording after playing the sweep')
    parser.add_argument('--fading', type=float, default=0.,
            help='length in seconds of the fade in and out of the sweep (default 0.)')
    parser.add_argument('--dev-in', type=int, default=None,
            help='input device number')
    parser.add_argument('--dev-out', type=int, default=None,
            help='output device number')
    parser.add_argument('--ch-in', dest='channels_input_mapping', action='append', type=int,
            help='List of channel numbers (starting with 1) to record.')
    parser.add_argument('--ch-out', dest='channels_output_mapping', action='append', type=int,
            help='List of channel numbers (starting with 1) where the columns of data shall be played back on. Must have the same length as number of channels in data (except if data is mono, in which case the signal is played back on all given output channels). Each channel number may only appear once in mapping.')
    parser.add_argument('--asc', action='store_true',
            help='wether the sweep is from high to low (default) or low to high frequencies')
    parser.add_argument('--deconv', action='store_true',
            help='if True, apply deconvolution to the recorded signal to remove the sweep (default False)')
    parser.add_argument('-p', '--plot', action='store_true',
            help='plot the resulting signal')

    args = parser.parse_args()

    if args.type not in _sweep_types:
        raise ValueError('Sweep must be exponential or linear')

    kwargs = dict(
            sweep_length=args.length,
            sweep_type=args.type,
            fs=args.fs,
            f_lo=args.f_lo,
            f_hi=args.f_hi,
            volume=args.volume,
            pre_delay=args.pre_delay,
            post_delay=args.post_delay,
            fade_in_out=args.fading,
            dev_in=args.dev_in,
            dev_out=args.dev_out,
            channels_input_mapping=args.channels_input_mapping,
            channels_output_mapping=args.channels_output_mapping,
            ascending=args.asc,
            deconvolution=args.deconv,
            plot=args.plot,
            )

    signals = measure_ir(**kwargs)

    if args.file is not None:
        from scipy.io import wavfile
        if args.deconv:
            wavfile.write(args.file + '.wav', args.fs, signals[0])
            wavfile.write(args.file + '.ir.wav', args.fs, signals[1])
        else:
            wavfile.write(args.file + '.wav', args.fs, signals)

