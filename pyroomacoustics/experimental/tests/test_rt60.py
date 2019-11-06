# Copyright (c) 2019 Robin Scheibler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Test code for RT60 measurement routine
"""
import numpy as np
import pyroomacoustics as pra

eps = 1e-15

room = pra.ShoeBox([10, 7, 6], fs=16000, absorption=0.35, max_order=17)
room.add_source([3, 2.5, 1.7])
room.add_microphone_array(pra.MicrophoneArray(np.array([[7, 3.7, 1.1]]).T, room.fs))
room.compute_rir()

ir = room.rir[0][0]


def test_rt60():
    """
    Very basic test that runs the function and checks that the value
    returned with different sampling frequencies are correct.
    """

    t60_samples = pra.experimental.measure_rt60(ir)
    t60_s = pra.experimental.measure_rt60(ir, fs=room.fs)

    assert abs(t60_s - t60_samples / room.fs) < eps

    t30_samples = pra.experimental.measure_rt60(ir, decay_db=30)
    t30_s = pra.experimental.measure_rt60(ir, decay_db=30, fs=room.fs)

    assert abs(t30_s - t30_samples / room.fs) < eps


def test_rt60_plot():
    """
    Simple run of the plot without any output.

    Check for runtime errors only.
    """

    import matplotlib
    matplotlib.use('Agg')

    pra.experimental.measure_rt60(ir, plot=True)
    pra.experimental.measure_rt60(ir, plot=True, rt60_tgt=0.3)


if __name__ == "__main__":

    test_rt60()
    test_rt60_plot()
