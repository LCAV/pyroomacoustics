'''
Tests the mixing function of ``Room.simulate``
'''
import numpy as np
import pyroomacoustics as pra
import unittest

room = pra.ShoeBox([9,5,4], fs=16000, absorption=0.25, max_order=15)

# three microphones
room.add_microphone_array(
        pra.MicrophoneArray(
            np.c_[
                [4.3, 2.1, 1.8],
                [2.5, 1.7, 1.2],
                [6.5, 3.9, 0.9],
                ],
            room.fs,
            )
        )

# add two sources (1 target, 1 interferer)
room.add_source([3.1, 1.5, 2.1], signal=np.random.randn(room.fs * 5))
room.add_source([5.8, 3.3, 3.1], signal=np.random.randn(room.fs * 3), delay=1.)

# the extra arguments are given in a dictionary
mix_kwargs = {
        'snr' : 30,  # SNR target is 30 decibels
        'sir' : 10,  # SIR target is 10 decibels
        'n_src' : 2,
        'n_tgt' : 1,
        'ref_mic' : 1,
        }

def callback_mix(premix, snr=0, sir=0, ref_mic=0, n_src=None, n_tgt=None):

    # first normalize all separate recording to have unit power at microphone one
    p_mic_ref = np.std(premix[:,ref_mic,:], axis=1)
    premix /= p_mic_ref[:,None,None]

    # now compute the power of interference signal needed to achieve desired SIR
    sigma_i = np.sqrt(10 ** (- sir / 10) / (n_src - n_tgt))
    premix[n_tgt:n_src,:,:] *= sigma_i

    # compute noise variance
    sigma_n = np.sqrt(10 ** (- snr / 10))

    # Mix down the recorded signals
    mix = np.sum(premix[:n_src,:], axis=0) + sigma_n * np.random.randn(*premix.shape[1:])

    return mix


class TestRoomMix(unittest.TestCase):

    def test_mix(self):

        # Run the simulation
        premix = room.simulate(
                callback_mix=callback_mix,
                callback_mix_kwargs=mix_kwargs,
                return_premix=True,
                )
        mics_signals = room.mic_array.signals

        # recover the noise signal
        i_ref = mix_kwargs['ref_mic']
        noise = mics_signals[i_ref,:] - np.sum(premix[:,i_ref,:], axis=0)
        noise_pwr = np.var(noise)

        premix_ref_pwr = np.var(premix[:,i_ref,:], axis=1)

        tgt_pwr = np.sum(premix_ref_pwr[:mix_kwargs['n_tgt']])
        int_pwr = np.sum(premix_ref_pwr[mix_kwargs['n_tgt']:])
        snr = 10 * np.log10(1. / noise_pwr)
        sir = 10 * np.log10(tgt_pwr / int_pwr)

        print('SNR', snr)
        print('SIR', sir)

        self.assertTrue(all([
            abs(snr - mix_kwargs['snr']) < 5e-2,
            abs(sir - mix_kwargs['sir']) < 5e-2,
            ]))


if __name__ == '__main__':
    unittest.main()

