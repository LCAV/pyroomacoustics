import numpy as np
import pyroomacoustics as pra

def test_soundsource_basic():

    the_signal = np.ones(10)
    source = pra.SoundSource([1., 1.], signal=the_signal)

    the_signal_2 = np.zeros(10)
    source.add_signal(the_signal_2)

if __name__ == '__main__':
    
    test_soundsource_basic()

