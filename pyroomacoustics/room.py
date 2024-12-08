# Main Room class using to encapsulate the room acoustics simulator
# Copyright (C) 2019  Robin Scheibler, Ivan Dokmanic, Sidney Barthe, Cyril Cadoux
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
#
# You should have received a copy of the MIT License along with this program. If
# not, see <https://opensource.org/licenses/MIT>.

r"""
Room
====

The three main classes are :py:obj:`pyroomacoustics.room.Room`,
:py:obj:`pyroomacoustics.soundsource.SoundSource`, and
:py:obj:`pyroomacoustics.beamforming.MicrophoneArray`. On a high level, a
simulation scenario is created by first defining a room to which a few sound
sources and a microphone array are attached. The actual audio is attached to
the source as raw audio samples.

Then, a simulation method is used to create artificial room impulse responses
(RIR) between the sources and microphones. The current default method is the
image source which considers the walls as perfect reflectors. An experimental
hybrid simulator based on image source method (ISM) [1]_ and ray tracing (RT) [2]_, [3]_, is also available.  Ray tracing
better capture the later reflections and can also model effects such as
scattering.

The microphone signals are then created by convolving audio samples associated
to sources with the appropriate RIR. Since the simulation is done on
discrete-time signals, a sampling frequency is specified for the room and the
sources it contains. Microphones can optionally operate at a different sampling
frequency; a rate conversion is done in this case.

Simulating a Shoebox Room with the Image Source Model
-----------------------------------------------------

We will first walk through the steps to simulate a shoebox-shaped room in 3D.
We use the ISM is to find all image sources up to a maximum specified order and
room impulse responses (RIR) are generated from their positions.

The code for the full example can be found in `examples/room_from_rt60.py`.

Create the room
~~~~~~~~~~~~~~~

So-called shoebox rooms are pallelepipedic rooms with 4 or 6 walls (in 2D and
3D respectiely), all at right angles. They are defined by a single vector that
contains the lengths of the walls. They have the advantage of being simple to
define and very efficient to simulate. In the following example, we define a
``9m x 7.5m x 3.5m`` room. In addition, we use `Sabine's formula <https://en.wikipedia.org/wiki/Reverberation>`_
to find the wall energy absorption and maximum order of the ISM required
to achieve a desired reverberation time (*RT60*, i.e. the time it takes for
the RIR to decays by 60 dB).

.. code-block:: python

    import pyroomacoustics as pra

    # The desired reverberation time and dimensions of the room
    rt60 = 0.5  # seconds
    room_dim = [9, 7.5, 3.5]  # meters

    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

    # Create the room
    room = pra.ShoeBox(
        room_dim, fs=16000, materials=pra.Material(e_absorption), max_order=max_order
    )

The second argument is the sampling frequency at which the RIR will be
generated. Note that the default value of ``fs`` is 8 kHz.
The third argument is the material of the wall, that itself takes the absorption as a parameter.
The fourth and last argument is the maximum number of reflections allowed in the ISM.

.. note::

    Note that Sabine's formula is only an approximation and that the actually
    simulated RT60 may vary by quite a bit.

.. warning::

    Until recently, rooms would take an ``absorption`` parameter that was
    actually **not** the energy absorption we use now.  The ``absorption``
    parameter is now deprecated and will be removed in the future.



Randomized Image Method
~~~~~~~~~~~~~~~~~~~~~~~~~


In highly symmetric shoebox rooms, the regularity of image sources’ positions
leads to a monotonic convergence in the time arrival of far-field image pairs.
This causes sweeping echoes. The randomized image method adds a small random
displacement to the image source positions, so that they are no longer
time-aligned, thus reducing sweeping echoes considerably.
To use the randomized method, set the flag ``use_rand_ism`` to True while creating
a room. Additionally, the maximum displacement of the image sources can be
chosen by setting the parameter ``max_rand_disp``. The default value is 8cm.
For a full example see examples/randomized_image_method.py

.. code-block:: python

    import pyroomacoustics as pra

    # The desired reverberation time and dimensions of the room
    rt60 = 0.5  # seconds
    room_dim = [5, 5, 5]  # meters

    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

    # Create the room
    room = pra.ShoeBox(
        room_dim, fs=16000, materials=pra.Material(e_absorption), max_order=max_order,
        use_rand_ism = True, max_rand_disp = 0.05
    )

Add sources and microphones
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sources are fairly straightforward to create. They take their location as single
mandatory argument, and a signal and start time as optional arguments.  Here we
create a source located at ``[2.5, 3.73, 1.76]`` within the room, that will utter
the content of the wav file ``speech.wav`` starting at ``1.3 s`` into the
simulation.  The ``signal`` keyword argument to the
:py:func:`~pyroomacoustics.room.Room.add_source` method should be a
one-dimensional ``numpy.ndarray`` containing the desired sound signal.

.. code-block:: python

    # import a mono wavfile as the source signal
    # the sampling frequency should match that of the room
    from scipy.io import wavfile
    _, audio = wavfile.read('speech.wav')

    # place the source in the room
    room.add_source([2.5, 3.73, 1.76], signal=audio, delay=1.3)

The locations of the microphones in the array should be provided in a numpy
``nd-array`` of size ``(ndim, nmics)``, that is each column contains the
coordinates of one microphone. Note that it can be different from that
of the room, in which case resampling will occur. Here, we create an array
with two microphones placed at ``[6.3, 4.87, 1.2]`` and ``[6.3, 4.93, 1.2]``.

.. code-block:: python

    # define the locations of the microphones
    import numpy as np
    mic_locs = np.c_[
        [6.3, 4.87, 1.2],  # mic 1
        [6.3, 4.93, 1.2],  # mic 2
    ]

    # finally place the array in the room
    room.add_microphone_array(mic_locs)

A number of routines exist to create regular array geometries in 2D.

- :py:func:`~pyroomacoustics.beamforming.linear_2D_array`
- :py:func:`~pyroomacoustics.beamforming.circular_2D_array`
- :py:func:`~pyroomacoustics.beamforming.square_2D_array`
- :py:func:`~pyroomacoustics.beamforming.poisson_2D_array`
- :py:func:`~pyroomacoustics.beamforming.spiral_2D_array`


Adding source or microphone directivity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The directivity pattern of a source or microphone can be conveniently set
through the ``directivity`` keyword argument.

First, a :py:obj:`pyroomacoustics.directivities.Directivity` object needs to be created. As of
Sep 6, 2021, only frequency-independent directivities from the
`cardioid family <https://en.wikipedia.org/wiki/Microphone#Cardioid,_hypercardioid,_supercardioid,_subcardioid>`_
are supported, namely figure-eight, hypercardioid, cardioid, and subcardioid.

Below is how a :py:obj:`pyroomacoustics.directivities.Directivity` object can be created, for
example a hypercardioid pattern pointing at an azimuth angle of 90 degrees and a colatitude
angle of 15 degrees.

.. code-block:: python

    # create directivity object
    from pyroomacoustics.directivities import (
        DirectivityPattern,
        DirectionVector,
        CardioidFamily,
    )
    dir_obj = CardioidFamily(
        orientation=DirectionVector(azimuth=90, colatitude=15, degrees=True),
        pattern_enum=DirectivityPattern.HYPERCARDIOID,
    )

After creating a :py:obj:`pyroomacoustics.directivities.Directivity` object, it is straightforward
to set the directivity of a source, microphone, or microphone array, namely by using the
``directivity`` keyword argument.

For example, to set a source's directivity:

.. code-block:: python

    # place the source in the room
    room.add_source(position=[2.5, 3.73, 1.76], directivity=dir_obj)

To set a single microphone's directivity:

.. code-block:: python

    # place the microphone in the room
    room.add_microphone(loc=[2.5, 5, 1.76], directivity=dir_obj)

The same directivity pattern can be used for all microphones in an array:

.. code-block:: python

    # place microphone array in the room
    import numpy as np
    mic_locs = np.c_[
        [6.3, 4.87, 1.2],  # mic 1
        [6.3, 4.93, 1.2],  # mic 2
    ]
    room.add_microphone_array(mic_locs, directivity=dir_obj)

Or a different directivity can be used for each microphone by passing a list of
:py:obj:`pyroomacoustics.directivities.Directivity` objects:

.. code-block:: python

    # place the microphone array in the room
    room.add_microphone_array(mic_locs, directivity=[dir_1, dir_2])

.. warning::

    As of Sep 6, 2021, setting directivity patterns for sources and microphone is only supported for
    the image source method (ISM). Moreover, source direcitivities are only supported for
    shoebox-shaped rooms.


Create the Room Impulse Response
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At this point, the RIRs are simply created by invoking the ISM via
:py:func:`~pyroomacoustics.room.Room.image_source_model`. This function will
generate all the images sources up to the order required and use them to
generate the RIRs, which will be stored in the ``rir`` attribute of ``room``.
The attribute ``rir`` is a list of lists so that the outer list is on microphones
and the inner list over sources.

.. code-block:: python

    room.compute_rir()

    # plot the RIR between mic 1 and source 0
    import matplotlib.pyplot as plt
    plt.plot(room.rir[1][0])
    plt.show()

.. warning::

    The simulator uses a fractional delay filter that introduce a global delay
    in the RIR. The delay can be obtained as follows.

    .. code-block:: python

        global_delay = pra.constants.get("frac_delay_length") // 2


Simulate sound propagation
~~~~~~~~~~~~~~~~~~~~~~~~~~

By calling :py:func:`~pyroomacoustics.room.Room.simulate`, a convolution of the
signal of each source (if not ``None``) will be performed with the
corresponding room impulse response. The output from the convolutions will be summed up
at the microphones. The result is stored in the ``signals`` attribute of ``room.mic_array``
with each row corresponding to one microphone.

.. code-block:: python

    room.simulate()

    # plot signal at microphone 1
    plt.plot(room.mic_array.signals[1,:])

Full Example
~~~~~~~~~~~~

This example is partly exctracted from `./examples/room_from_rt60.py`.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import pyroomacoustics as pra
    from scipy.io import wavfile

    # The desired reverberation time and dimensions of the room
    rt60_tgt = 0.3  # seconds
    room_dim = [10, 7.5, 3.5]  # meters

    # import a mono wavfile as the source signal
    # the sampling frequency should match that of the room
    fs, audio = wavfile.read("examples/samples/guitar_16k.wav")

    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

    # Create the room
    room = pra.ShoeBox(
        room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
    )

    # place the source in the room
    room.add_source([2.5, 3.73, 1.76], signal=audio, delay=0.5)

    # define the locations of the microphones
    mic_locs = np.c_[
        [6.3, 4.87, 1.2], [6.3, 4.93, 1.2],  # mic 1  # mic 2
    ]

    # finally place the array in the room
    room.add_microphone_array(mic_locs)

    # Run the simulation (this will also build the RIR automatically)
    room.simulate()

    room.mic_array.to_wav(
        f"examples/samples/guitar_16k_reverb_{args.method}.wav",
        norm=True,
        bitdepth=np.int16,
    )

    # measure the reverberation time
    rt60 = room.measure_rt60()
    print("The desired RT60 was {}".format(rt60_tgt))
    print("The measured RT60 is {}".format(rt60[1, 0]))

    # Create a plot
    plt.figure()

    # plot one of the RIR. both can also be plotted using room.plot_rir()
    rir_1_0 = room.rir[1][0]
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(rir_1_0)) / room.fs, rir_1_0)
    plt.title("The RIR from source 0 to mic 1")
    plt.xlabel("Time [s]")

    # plot signal at microphone 1
    plt.subplot(2, 1, 2)
    plt.plot(room.mic_array.signals[1, :])
    plt.title("Microphone 1 signal")
    plt.xlabel("Time [s]")

    plt.tight_layout()
    plt.show()




Hybrid ISM/Ray Tracing Simulator
--------------------------------

.. warning::

    The hybrid simulator has not been thoroughly tested yet and should be used with
    care. The exact implementation and default settings may also change in the future.
    Currently, the default behavior of :py:obj:`~pyroomacoustics.room.Room`
    and :py:obj:`~pyroomacoustics.room.ShoeBox` has been kept as in previous
    versions of the package. Bugs and user experience can be reported on
    `github <https://github.com/LCAV/pyroomacoustics>`_.

The hybrid ISM/RT simulator uses ISM to simulate the early reflections in the RIR
and RT for the diffuse tail. Our implementation is based on [2]_ and [3]_.

The simulator has the following features.

- Scattering: Wall scattering can be defined by assigning a scattering
  coefficient to the walls together with the energy absorption.
- Multi-band: The simulation can be carried out with different parameters for
  different `octave bands <https://en.wikipedia.org/wiki/Octave_band>`_. The
  octave bands go from 125 Hz to half the sampling frequency.
- Air absorption: The frequency dependent absorption of the air can be turned
  by providing the keyword argument ``air_absorption=True`` to the room
  constructor.

Here is a simple example using the hybrid simulator.
We suggest to use ``max_order=3`` with the hybrid simulator.

.. code-block:: python

    # Create the room
    room = pra.ShoeBox(
        room_dim,
        fs=16000,
        materials=pra.Material(e_absorption),
        max_order=3,
        ray_tracing=True,
        air_absorption=True,
    )

    # Activate the ray tracing
    room.set_ray_tracing()

A few example programs are provided in ``./examples``.

- ``./examples/ray_tracing.py`` demonstrates use of ray tracing for rooms of different sizes
  and with different amounts of reverberation
- ``./examples/room_L_shape_3d_rt.py`` shows how to simulate a polyhedral room
- ``./examples/room_from_stl.py`` demonstrates how to import a model from an STL file



Wall Materials
--------------

The wall materials are handled by the
:py:obj:`~pyroomacoustics.parameters.Material` objects.  A material is defined
by at least one *absorption* coefficient that represents the ratio of sound
energy absorbed by a wall upon reflection.
A material may have multiple absorption coefficients corresponding to different
abosrptions at different octave bands.
When only one coefficient is provided, the absorption is assumed to be uniform at
all frequencies.
In addition, materials may have one or more scattering coefficients
corresponding to the ratio of energy scattered upon reflection.

The materials can be defined by providing the coefficients directly, or they can
be defined by chosing a material from the :doc:`materials database<pyroomacoustics.materials.database>` [2]_.

.. code-block:: python

    import pyroomacoustics as pra
    m = pra.Material(energy_absorption="hard_surface")
    room = pra.ShoeBox([9, 7.5, 3.5], fs=16000, materials=m, max_order=17)

We can use different materials for different walls. In this case, the materials should be
provided in a dictionary. For a shoebox room, this can be done as follows.
We use the :py:func:`~pyroomacoustics.parameters.make_materials` helper
function to create a ``dict`` of
:py:class:`~pyroomacoustics.parameters.Material` objects.

.. code-block:: python

    import pyroomacoustics as pra
    m = pra.make_materials(
        ceiling="hard_surface",
        floor="6mm_carpet",
        east="brickwork",
        west="brickwork",
        north="brickwork",
        south="brickwork",
    )
    room = pra.ShoeBox(
        [9, 7.5, 3.5], fs=16000, materials=m, max_order=17, air_absorption=True, ray_tracing=True
    )

For a more complete example see
`examples/room_complex_wall_materials.py
<https://github.com/LCAV/pyroomacoustics/blob/master/examples/room_complex_wall_materials.py>`_.

.. note::

    For shoebox rooms, the walls are labelled as follows:

    - ``west``/``east`` for the walls in the y-z plane with a small/large x coordinates, respectively
    - ``south``/``north`` for the walls in the x-z plane with a small/large y coordinates, respectively
    - ``floor``/``ceiling`` for the walls int x-y plane with small/large z coordinates, respectively


Air Absorption
--------------

The absorption of sound energy by air is frequency dependent.
The absorption is described the frequency dependent coefficient :math:`\\alpha(f)` and the energy decreases with distance as :math:`e^{-\\alpha(f) d}`.
This can be turned simply by providing the keyword argument ``air_absorption=True`` to the room constructor or calling ``set_absorption()`` on an existing room.
The coefficients are also temperature and humidity dependent and the default values are as follows.

========= ========    ====== ====== ====== ===== ===== ===== ===== =====
Temp. (C) Hum. (%)    125 Hz 250 Hz 500 Hz 1 kHz 2 kHz 4 kHz 8 kHz 
========= ========    ====== ====== ====== ===== ===== ===== ===== =====
10        30-50       0.1    0.2    0.5    1.1   2.7   9.4   29.0  x1e-3
10        50-70       0.1    0.2    0.5    0.8   1.8   5.9   21.1  x1e-3
10        70-90       0.1    0.2    0.5    0.7   1.4   4.4   15.8  x1e-3
20        30-50       0.1    0.3    0.6    1.0   1.9   5.8   20.3  x1e-3
20        50-70       0.1    0.3    0.6    1.0   1.7   4.1   13.5  x1e-3
20        70-90       0.1    0.3    0.6    1.1   1.7   3.5   10.6  x1e-3
========= ========    ====== ====== ====== ===== ===== ===== ===== =====

It is possible to set custom coefficients by providing a lists of coefficients and corresponding frequencies.
If the frequencies are not provided, the default values of 125 Hz to 8 kHz octave bands are assumed.
Note, that the number of octave bands will depend on the sampling frequency used.
For 16 kHz, there will be 7 octave bands.
If less than 7 coefficients are provided, or if the center frequencies do not correspond, a simple interpolation is used to fill the missing values.
Missing values at end of the array are simply replicated from the last value.

.. code-block:: python

    room.set_air_absorption([0.1, 0.2, 0.4, 1.3, 2.8, 9.4, 23.0])

Controlling the signal-to-noise ratio
-------------------------------------

It is in general necessary to scale the signals from different sources to
obtain a specific signal-to-noise or signal-to-interference ratio (SNR and SIR,
respectively). This can be done by passing some options to the :py:func:`simulate()`
function. Because the relative amplitude of signals will change at different microphones
due to propagation, it is necessary to choose a reference microphone. By default, this
will be the first microphone in the array (index 0). The simplest choice is to choose
the variance of the noise \\(\\sigma_n^2\\) to achieve a desired SNR with respect
to the cumulative signal from all sources. Assuming that the signals from all sources
are scaled to have the same amplitude (e.g., unit amplitude) at the reference microphone,
the SNR is defined as

.. math::

    \mathsf{SNR} = 10 \log_{10} \frac{K}{\sigma_n^2}

where \\(K\\) is the number of sources. For example, an SNR of 10 decibels (dB)
can be obtained using the following code

.. code-block:: python

    room.simulate(reference_mic=0, snr=10)

Sometimes, more challenging normalizations are necessary. In that case,
a custom callback function can be provided to simulate. For example,
we can imagine a scenario where we have ``n_src`` out of which ``n_tgt``
are the targets, the rest being interferers. We will assume all
targets have unit variance, and all interferers have equal
variance \\(\\sigma_i^2\\) (at the reference microphone). In
addition, there is uncorrelated noise \\(\\sigma_n^2\\) at
every microphones. We will define SNR and SIR with respect
to a single target source:

.. math::

    \mathsf{SNR} & = 10 \log_{10} \frac{1}{\sigma_n^2}

    \mathsf{SIR} & = 10 \log_{10} \frac{1}{(\mathsf{n_{src}} - \mathsf{n_{tgt}}) \sigma_i^2}

The callback function ``callback_mix`` takes as argument an nd-array
``premix_signals`` of shape ``(n_src, n_mics, n_samples)`` that contains the
microphone signals prior to mixing. The signal propagated from the ``k``-th
source to the ``m``-th microphone is contained in ``premix_signals[k,m,:]``. It
is possible to provide optional arguments to the callback via
``callback_mix_kwargs`` optional argument. Here is the code
implementing the example described.

.. code-block:: python

    # the extra arguments are given in a dictionary
    callback_mix_kwargs = {
            'snr' : 30,  # SNR target is 30 decibels
            'sir' : 10,  # SIR target is 10 decibels
            'n_src' : 6,
            'n_tgt' : 2,
            'ref_mic' : 0,
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

    # Run the simulation
    room.simulate(
            callback_mix=callback_mix,
            callback_mix_kwargs=callback_mix_kwargs,
            )
    mics_signals = room.mic_array.signals

In addition, it is desirable in some cases to obtain the microphone signals
with individual sources, prior to mixing. For example, this is useful to
evaluate the output from blind source separation algorithms. In this case, the
``return_premix`` argument should be set to ``True``

.. code-block:: python

    premix = room.simulate(return_premix=True)


Reverberation Time
------------------

The reverberation time (RT60) is defined as the time needed for the enery of
the RIR to decrease by 60 dB. It is a useful measure of the amount of
reverberation.  We provide a method in the
:py:func:`~pyroomacoustics.experimental.rt60.measure_rt60` to measure the RT60
of recorded or simulated RIR.

The method is also directly integrated in the :py:obj:`~pyroomacoustics.room.Room` object as the method :py:func:`~pyroomacoustics.room.Room.measure_rt60`.

.. code-block:: python

    # assuming the simulation has already been carried out
    rt60 = room.measure_rt60()

    for m in room.n_mics:
        for s in room.n_sources:
            print(
                "RT60 between the {}th mic and {}th source: {:.3f} s".format(m, s, rt60[m, s])
            )

Free-field simulation
=====================

You can also use this package to simulate free-field sound propagation between
a set of sound sources and a set of microphones, without considering room
effects. To this end, you can use the
:py:obj:`pyroomacoustics.room.AnechoicRoom` class, which simply corresponds to
setting the maximum image image order of the room simulation to zero. This
allows for early development and testing of various audio-based algorithms,
without worrying about room acoustics at first. Thanks to the modular framework
of pyroomacoustics, room acoustics can easily be added, after this first
testing stage, for more realistic simulations. 

Use this if you can neglect room effects (e.g. you operate in an anechoic room
or outdoors), or if you simply want to test your algorithm in the best-case
scenario. The below code shows how to create and simualte an anechoic room. For
a more involved example (testing a the DOA algorithm MUSIC in an anechoic
room), see `./examples/doa_anechoic_room.py`.

.. code-block:: python

    # Create anechoic room. 
    room = pra.AnechoicRoom(fs=16000)

    # Place the microphone array around the origin.
    mic_locs = np.c_[
        [0.1, 0.1, 0],
        [-0.1, 0.1, 0],
        [-0.1, -0.1, 0],
        [0.1, -0.1, 0],
    ]
    room.add_microphone_array(mic_locs)

    # Add a source. We use a white noise signal for the source, and
    # the source can be arbitrarily far because there are no walls.
    x = np.random.randn(2**10)
    room.add_source([10.0, 20.0, -20], signal=x)

    # run the simulation
    room.simulate()

References
----------

.. [1] J. B. Allen and D. A. Berkley, *Image method for efficiently simulating small-room acoustics,* J. Acoust. Soc. Am., vol. 65, no. 4, p. 943, 1979.

.. [2] M. Vorlaender, Auralization, 1st ed. Berlin: Springer-Verlag, 2008, pp. 1-340.

.. [3] D. Schroeder, Physically based real-time auralization of interactive virtual environments. PhD Thesis, RWTH Aachen University, 2011.

"""


from __future__ import division, print_function

import math
import os
import sys
import warnings

import numpy as np
import scipy.spatial as spatial
from scipy.interpolate import interp1d

from . import beamforming as bf
from . import libroom
from .acoustics import OctaveBandsFactory, rt60_eyring, rt60_sabine
from .beamforming import MicrophoneArray
from .directivities import CardioidFamily, MeasuredDirectivity
from .experimental import measure_rt60
from .libroom import Wall, Wall2D
from .parameters import Material, Physics, constants, eps, make_materials
from .simulation import compute_ism_rir, compute_rt_rir
from .soundsource import SoundSource


def wall_factory(corners, absorption, scattering, name=""):
    """Call the correct method according to wall dimension"""
    if corners.shape[0] == 3:
        return Wall(corners, absorption, scattering, name)
    elif corners.shape[0] == 2:
        return Wall2D(corners, absorption, scattering, name)
    else:
        raise ValueError("Rooms can only be 2D or 3D")


def find_non_convex_walls(walls):
    """
    Finds the walls that are not in the convex hull

    Parameters
    ----------
    walls: list of Wall objects
        The walls that compose the room

    Returns
    -------
    list of int
        The indices of the walls no in the convex hull
    """

    all_corners = []
    for wall in walls[1:]:
        all_corners.append(wall.corners.T)
    X = np.concatenate(all_corners, axis=0)
    convex_hull = spatial.ConvexHull(X, incremental=True)

    # Now we need to check which walls are on the surface
    # of the hull
    in_convex_hull = [False] * len(walls)
    for i, wall in enumerate(walls):
        # We check if the center of the wall is co-linear or co-planar
        # with a face of the convex hull
        point = np.mean(wall.corners, axis=1)

        for simplex in convex_hull.simplices:
            if point.shape[0] == 2:
                # check if co-linear
                p0 = convex_hull.points[simplex[0]]
                p1 = convex_hull.points[simplex[1]]
                if libroom.ccw3p(p0, p1, point) == 0:
                    # co-linear point add to hull
                    in_convex_hull[i] = True

            elif point.shape[0] == 3:
                # Check if co-planar
                p0 = convex_hull.points[simplex[0]]
                p1 = convex_hull.points[simplex[1]]
                p2 = convex_hull.points[simplex[2]]

                normal = np.cross(p1 - p0, p2 - p0)
                if np.abs(np.inner(normal, point - p0)) < eps:
                    # co-planar point found!
                    in_convex_hull[i] = True

    return [i for i in range(len(walls)) if not in_convex_hull[i]]


class Room(object):
    """
    A Room object has as attributes a collection of
    :py:obj:`pyroomacoustics.wall.Wall` objects, a
    :py:obj:`pyroomacoustics.beamforming.MicrophoneArray` array, and a list of
    :py:obj:`pyroomacoustics.soundsource.SoundSource`. The room can be two
    dimensional (2D), in which case the walls are simply line segments. A factory method
    :py:func:`pyroomacoustics.room.Room.from_corners`
    can be used to create the room from a polygon. In three dimensions (3D), the
    walls are two dimensional polygons, namely a collection of points lying on a
    common plane. Creating rooms in 3D is more tedious and for convenience a method
    :py:func:`pyroomacoustics.room.Room.extrude` is provided to lift a 2D room
    into 3D space by adding vertical walls and parallel floor and ceiling.

    The Room is sub-classed by :py:obj:`pyroomacoustics.room.ShoeBox` which
    creates a rectangular (2D) or parallelepipedic (3D) room. Such rooms
    benefit from an efficient algorithm for the image source method.


    :attribute walls: (Wall array) list of walls forming the room
    :attribute fs: (int) sampling frequency
    :attribute max_order: (int) the maximum computed order for images
    :attribute sources: (SoundSource array) list of sound sources
    :attribute mics: (MicrophoneArray) array of microphones
    :attribute corners: (numpy.ndarray 2xN or 3xN, N=number of walls) array containing a point belonging to each wall, used for calculations
    :attribute absorption: (numpy.ndarray size N, N=number of walls)  array containing the absorption factor for each wall, used for calculations
    :attribute dim: (int) dimension of the room (2 or 3 meaning 2D or 3D)
    :attribute wallsId: (int dictionary) stores the mapping "wall name -> wall id (in the array walls)"

    Parameters
    ----------
    walls: list of Wall or Wall2D objects
        The walls forming the room.
    fs: int, optional
        The sampling frequency in Hz. Default is 8000.
    t0: float, optional
        The global starting time of the simulation in seconds. Default is 0.
    max_order: int, optional
        The maximum reflection order in the image source model. Default is 1,
        namely direct sound and first order reflections.
    sigma2_awgn: float, optional
        The variance of the additive white Gaussian noise added during
        simulation. By default, none is added.
    sources: list of SoundSource objects, optional
        Sources to place in the room. Sources can be added after room creating
        with the `add_source` method by providing coordinates.
    mics: MicrophoneArray object, optional
        The microphone array to place in the room. A single microphone or
        microphone array can be added after room creation with the
        `add_microphone_array` method.
    temperature: float, optional
        The air temperature in the room in degree Celsius. By default, set so
        that speed of sound is 343 m/s.
    humidity: float, optional
        The relative humidity of the air in the room (between 0 and 100). By
        default set to 0.
    air_absorption: bool, optional
        If set to True, absorption of sound energy by the air will be
        simulated.
    ray_tracing: bool, optional
        If set to True, the ray tracing simulator will be used along with
        image source model.
    use_rand_ism: bool, optional
        If set to True, image source positions will have a small random
        displacement to prevent sweeping echoes
    max_rand_disp: float, optional;
        If using randomized image source method, what is the maximum
        displacement of the image sources?
    min_phase: bool, optional
        If set to True, generated RIRs will have a minimum phase response.
        Cannot be used with ray tracing model.
    """

    def __init__(
        self,
        walls,
        fs=8000,
        t0=0.0,
        max_order=1,
        sigma2_awgn=None,
        sources=None,
        mics=None,
        temperature=None,
        humidity=None,
        air_absorption=False,
        ray_tracing=False,
        use_rand_ism=False,
        max_rand_disp=0.08,
        min_phase=False,
    ):
        self.walls = walls

        # Get the room dimension from that of the walls
        self.dim = walls[0].dim

        # Create a mapping with friendly names for walls
        self._wall_mapping()

        # initialize everything else
        self._var_init(
            fs,
            t0,
            max_order,
            sigma2_awgn,
            temperature,
            humidity,
            air_absorption,
            ray_tracing,
            use_rand_ism,
            max_rand_disp,
            min_phase,
        )

        # initialize the C++ room engine
        self._init_room_engine()

        # add the sources
        self.sources = []
        if sources is not None and isinstance(sources, list):
            for src in sources:
                self.add_soundsource(src)

        # add the microphone array
        if mics is not None:
            self.add_microphone_array(mics)
        else:
            self.mic_array = None

    def _var_init(
        self,
        fs,
        t0,
        max_order,
        sigma2_awgn,
        temperature,
        humidity,
        air_absorption,
        ray_tracing,
        use_rand_ism,
        max_rand_disp,
        min_phase,
    ):
        self.fs = fs

        if t0 != 0.0:
            raise NotImplementedError(
                "Global simulation delay not " "implemented (aka t0)"
            )
        self.t0 = t0

        self.max_order = max_order
        self.sigma2_awgn = sigma2_awgn

        self.octave_bands = OctaveBandsFactory(
            fs=self.fs,
            n_fft=constants.get("octave_bands_n_fft"),
            keep_dc=constants.get("octave_bands_keep_dc"),
            base_frequency=constants.get("octave_bands_base_freq"),
        )
        self.max_rand_disp = max_rand_disp

        # Keep track of the state of the simulator
        self.simulator_state = {
            "ism_needed": (self.max_order >= 0),
            "random_ism_needed": use_rand_ism,
            "rt_needed": ray_tracing,
            "air_abs_needed": air_absorption,
            "ism_done": False,
            "rt_done": False,
            "rir_done": False,
        }

        # make it clear the room (C++) engine is not ready yet
        self.room_engine = None

        if temperature is None and humidity is None:
            # default to package wide setting when nothing is provided
            self.physics = Physics().from_speed(constants.get("c"))
        else:
            # use formulas when temperature and/or humidity are provided
            self.physics = Physics(temperature=temperature, humidity=humidity)

        self.set_sound_speed(self.physics.get_sound_speed())
        self.air_absorption = None
        if air_absorption:
            self.set_air_absorption()

        # default values for ray tracing parameters
        self._set_ray_tracing_options(use_ray_tracing=ray_tracing)

        # in the beginning, nothing has been
        self.visibility = None

        # initialize the attribute for the impulse responses
        self.rir = None

        self.min_phase = min_phase

    def _init_room_engine(self, *args):
        args = list(args)

        if len(args) == 0:
            # This is a polygonal room
            # find the non convex walls
            obstructing_walls = find_non_convex_walls(self.walls)
            args += [self.walls, obstructing_walls]

        # for shoebox rooms, the required arguments are passed to
        # the function

        # initialize the C++ room engine
        args += [
            [],
            self.c,  # speed of sound
            self.max_order,
            self.rt_args["energy_thres"],
            self.rt_args["time_thres"],
            self.rt_args["receiver_radius"],
            self.rt_args["hist_bin_size"],
            self.simulator_state["ism_needed"] and self.simulator_state["rt_needed"],
        ]

        # Create the real room object
        if self.dim == 2:
            self.room_engine = libroom.Room2D(*args)
        else:
            self.room_engine = libroom.Room(*args)

    def _update_room_engine_params(self):
        # Now, if it exists, set the parameters of room engine
        if self.room_engine is not None:
            self.room_engine.set_params(
                self.c,  # speed of sound
                self.max_order,
                self.rt_args["energy_thres"],
                self.rt_args["time_thres"],
                self.rt_args["receiver_radius"],
                self.rt_args["hist_bin_size"],
                (
                    self.simulator_state["ism_needed"]
                    and self.simulator_state["rt_needed"]
                ),
            )

    @property
    def is_multi_band(self):
        multi_band = False
        for w in self.walls:
            if len(w.absorption) > 1:
                multi_band = True
        return multi_band

    def set_ray_tracing(
        self,
        n_rays=None,
        receiver_radius=0.5,
        energy_thres=1e-7,
        time_thres=10.0,
        hist_bin_size=0.004,
    ):
        """
        Activates the ray tracer.

        Parameters
        ----------
        n_rays: int, optional
            The number of rays to shoot in the simulation
        receiver_radius: float, optional
            The radius of the sphere around the microphone in which to
            integrate the energy (default: 0.5 m)
        energy_thres: float, optional
            The energy thresold at which rays are stopped (default: 1e-7)
        time_thres: float, optional
            The maximum time of flight of rays (default: 10 s)
        hist_bin_size: float
            The time granularity of bins in the energy histogram (default: 4 ms)
        """
        self._set_ray_tracing_options(
            use_ray_tracing=True,
            n_rays=n_rays,
            receiver_radius=receiver_radius,
            energy_thres=energy_thres,
            time_thres=time_thres,
            hist_bin_size=hist_bin_size,
        )

    def _set_ray_tracing_options(
        self,
        use_ray_tracing,
        n_rays=None,
        receiver_radius=0.5,
        energy_thres=1e-7,
        time_thres=10.0,
        hist_bin_size=0.004,
        is_init=False,
    ):
        """
        Base method to set all ray tracing related options
        """

        if use_ray_tracing:
            if hasattr(self, "mic_array") and self.mic_array is not None:
                if self.mic_array.is_directive:
                    raise NotImplementedError(
                        "Directivity not supported with ray tracing."
                    )
            if hasattr(self, "sources"):
                for source in self.sources:
                    if source.directivity is not None:
                        raise NotImplementedError(
                            "Directivity not supported with ray tracing."
                        )

        self.simulator_state["rt_needed"] = use_ray_tracing

        self.rt_args = {}
        self.rt_args["energy_thres"] = energy_thres
        self.rt_args["time_thres"] = time_thres
        self.rt_args["receiver_radius"] = receiver_radius
        self.rt_args["hist_bin_size"] = hist_bin_size

        # set the histogram bin size so that it is an integer number of samples
        self.rt_args["hist_bin_size_samples"] = math.floor(
            self.fs * self.rt_args["hist_bin_size"]
        )
        self.rt_args["hist_bin_size"] = self.rt_args["hist_bin_size_samples"] / self.fs

        if n_rays is None:
            n_rays_auto_flag = True

            # We follow Vorlaender 2008, Eq. (11.12) to set the default number of rays
            # It depends on the mean hit rate we want to target
            target_mean_hit_count = 20

            # This is the multiplier for a single hit in average
            k1 = self.get_volume() / (
                np.pi
                * (self.rt_args["receiver_radius"] ** 2)
                * self.c
                * self.rt_args["hist_bin_size"]
            )

            n_rays = int(target_mean_hit_count * k1)

            if self.simulator_state["rt_needed"] and n_rays > 100000:
                import warnings

                warnings.warn(
                    "The number of rays used for ray tracing is larger than"
                    "100000 which may result in slow simulation.  The number"
                    "of rays was automatically chosen to provide accurate"
                    "room impulse response based on the room volume and the"
                    "receiver radius around the microphones.  The number of"
                    "rays may be reduced by increasing the size of the"
                    "receiver.  This tends to happen especially for large"
                    "rooms with small receivers.  The receiver is a sphere"
                    "around the microphone and its radius (in meters) may be"
                    "specified by providing the `receiver_radius` keyword"
                    "argument to the `set_ray_tracing` method."
                )

        self.rt_args["n_rays"] = n_rays

        self._update_room_engine_params()

    def unset_ray_tracing(self):
        """Deactivates the ray tracer"""
        self.simulator_state["rt_needed"] = False
        self._update_room_engine_params()

    def set_air_absorption(
        self, coefficients=None, center_freqs=None, interp_kind="linear"
    ):
        """
        Activates or deactivates air absorption in the simulation.

        Parameters
        ----------
        coefficients: list of float, optional
            Optional list of air absorption coefficients, one per octave band.
            If not provided, values corresponding to the temperature and humidity
            of the room are used.
        center_freqs: list, optional
            The optional list of center frequencies for the octave bands.
            If not provided, the values of the default ocatave bands are used.
        interp_kind: str
            Specifies the kind of interpolation as a string (‘linear’,
            ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’,
            ‘next’, where ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a
            spline interpolation of zeroth, first, second or third order;
            ‘previous’ and ‘next’ simply return the previous or next value of
            the point) or as an integer specifying the order of the spline
            interpolator to use. Default is ‘linear’.
        """

        self.simulator_state["air_abs_needed"] = True
        if coefficients is None:
            self.air_absorption = self.octave_bands(**self.physics.get_air_absorption())
        else:
            # ignore temperature and humidity if coefficients are provided
            self.air_absorption = self.octave_bands(
                coeffs=coefficients, center_freqs=center_freqs, interp_kind=interp_kind
            )

    def unset_air_absorption(self):
        """Deactivates air absorption in the simulation"""
        self.simulator_state["air_abs_needed"] = False

    def set_sound_speed(self, c):
        """Sets the speed of sound unconditionnaly"""
        self.c = c
        self._update_room_engine_params()

    def _wall_mapping(self):
        # mapping between wall names and indices
        self.wallsId = {}
        for i in range(len(self.walls)):
            if self.walls[i].name is not None:
                self.wallsId[self.walls[i].name] = i

    @classmethod
    def from_corners(
        cls,
        corners,
        absorption=None,
        fs=8000,
        t0=0.0,
        max_order=1,
        sigma2_awgn=None,
        sources=None,
        mics=None,
        materials=None,
        **kwargs,
    ):
        """
        Creates a 2D room by giving an array of corners.

        Parameters
        ----------
        corners: (np.array dim 2xN, N>2)
            list of corners, must be antiClockwise oriented
        absorption: float array or float
            list of absorption factor for each wall or single value
            for all walls (deprecated, use ``materials`` instead)
        fs: int, optional
            The sampling frequency in Hz. Default is 8000.
        t0: float, optional
            The global starting time of the simulation in seconds. Default is 0.
        max_order: int, optional
            The maximum reflection order in the image source model. Default is 1,
            namely direct sound and first order reflections.
        sigma2_awgn: float, optional
            The variance of the additive white Gaussian noise added during
            simulation. By default, none is added.
        sources: list of SoundSource objects, optional
            Sources to place in the room. Sources can be added after room creating
            with the `add_source` method by providing coordinates.
        mics: MicrophoneArray object, optional
            The microphone array to place in the room. A single microphone or
            microphone array can be added after room creation with the
            `add_microphone_array` method.
        kwargs: key, value mappings
            Other keyword arguments accepted by the :py:class:`~pyroomacoustics.room.Room` class

        Returns
        -------
        Instance of a 2D room
        """
        # make sure the corners are wrapped in an ndarray
        corners = np.array(corners)
        n_walls = corners.shape[1]

        corners = np.array(corners)
        if corners.shape[0] != 2 or n_walls < 3:
            raise ValueError("Arg corners must be more than two 2D points.")

        # We want to make sure the corners are ordered counter-clockwise
        if libroom.area_2d_polygon(corners) <= 0:
            corners = corners[:, ::-1]

        ############################
        # BEGIN COMPATIBILITY CODE #
        ############################

        if absorption is None:
            absorption = 0.0
            absorption_compatibility_request = False
        else:
            absorption_compatibility_request = True

        absorption = np.array(absorption, dtype="float64")
        if absorption.ndim == 0:
            absorption = absorption * np.ones(n_walls)
        elif absorption.ndim >= 1 and n_walls != len(absorption):
            raise ValueError(
                "Arg absorption must be the same size as corners or must be a single value."
            )

        ############################
        # BEGIN COMPATIBILITY CODE #
        ############################

        if materials is not None:
            if absorption_compatibility_request:
                import warnings

                warnings.warn(
                    "Because materials were specified, deprecated absorption parameter is ignored.",
                    DeprecationWarning,
                )

            if not isinstance(materials, list):
                materials = [materials] * n_walls

            if len(materials) != n_walls:
                raise ValueError("One material per wall is necessary.")

            for i in range(n_walls):
                assert isinstance(
                    materials[i], Material
                ), "Material not specified using correct class"

        elif absorption_compatibility_request:
            import warnings

            warnings.warn(
                "Using absorption parameter is deprecated. In the future, use materials instead."
            )

            # Fix the absorption
            # 1 - a1 == sqrt(1 - a2)    <-- a1 is former incorrect absorption, a2 is the correct definition based on energy
            # <=> a2 == 1 - (1 - a1) ** 2
            correct_absorption = 1.0 - (1.0 - absorption) ** 2
            materials = make_materials(*correct_absorption)

        else:
            # In this case, no material is provided, use totally reflective walls, no scattering
            materials = [Material(0.0, 0.0)] * n_walls

        # Resample material properties at octave bands
        octave_bands = OctaveBandsFactory(
            fs=fs,
            n_fft=constants.get("octave_bands_n_fft"),
            keep_dc=constants.get("octave_bands_keep_dc"),
            base_frequency=constants.get("octave_bands_base_freq"),
        )
        if not Material.all_flat(materials):
            for mat in materials:
                mat.resample(octave_bands)

        # Create the walls
        walls = []
        for i in range(n_walls):
            walls.append(
                wall_factory(
                    np.array([corners[:, i], corners[:, (i + 1) % n_walls]]).T,
                    materials[i].absorption_coeffs,
                    materials[i].scattering_coeffs,
                    "wall_" + str(i),
                )
            )

        return cls(
            walls,
            fs=fs,
            t0=t0,
            max_order=max_order,
            sigma2_awgn=sigma2_awgn,
            sources=sources,
            mics=mics,
            **kwargs,
        )

    def extrude(self, height, v_vec=None, absorption=None, materials=None):
        """
        Creates a 3D room by extruding a 2D polygon.
        The polygon is typically the floor of the room and will have z-coordinate zero. The ceiling

        Parameters
        ----------
        height : float
            The extrusion height
        v_vec : array-like 1D length 3, optional
            A unit vector. An orientation for the extrusion direction. The
            ceiling will be placed as a translation of the floor with respect
            to this vector (The default is [0,0,1]).
        absorption : float or array-like, optional
            Absorption coefficients for all the walls. If a scalar, then all the walls
            will have the same absorption. If an array is given, it should have as many elements
            as there will be walls, that is the number of vertices of the polygon plus two. The two
            last elements are for the floor and the ceiling, respectively.
            It is recommended to use materials instead of absorption parameter. (Default: 1)
        materials : dict
            Absorption coefficients for floor and ceiling. This parameter overrides absorption.
            (Default: {"floor": 1, "ceiling": 1})
        """

        if self.dim != 2:
            raise ValueError("Can only extrude a 2D room.")

        # default orientation vector is pointing up
        if v_vec is None:
            v_vec = np.array([0.0, 0.0, 1.0])

        # check that the walls are ordered counterclock wise
        # that should be the case if created from from_corners function
        nw = len(self.walls)
        floor_corners = np.zeros((2, nw))
        floor_corners[:, 0] = self.walls[0].corners[:, 0]
        ordered = True
        for iw, wall in enumerate(self.walls[1:]):
            if not np.allclose(self.walls[iw].corners[:, 1], wall.corners[:, 0]):
                ordered = False
            floor_corners[:, iw + 1] = wall.corners[:, 0]
        if not np.allclose(self.walls[-1].corners[:, 1], self.walls[0].corners[:, 0]):
            ordered = False

        if not ordered:
            raise ValueError(
                "The wall list should be ordered counter-clockwise, which is the case \
                if the room is created with Room.from_corners"
            )

        # make sure the floor_corners are ordered anti-clockwise (for now)
        if libroom.area_2d_polygon(floor_corners) <= 0:
            floor_corners = np.fliplr(floor_corners)

        wall_corners = {}
        wall_materials = {}
        for i in range(nw):
            name = str(i)
            wall_corners[name] = np.array(
                [
                    np.r_[floor_corners[:, i], 0],
                    np.r_[floor_corners[:, (i + 1) % nw], 0],
                    np.r_[floor_corners[:, (i + 1) % nw], 0] + height * v_vec,
                    np.r_[floor_corners[:, i], 0] + height * v_vec,
                ]
            ).T

            if len(self.walls[i].absorption) == 1:
                # Single band
                wall_materials[name] = Material(
                    energy_absorption=float(self.walls[i].absorption),
                    scattering=float(self.walls[i].scatter),
                )
            elif len(self.walls[i].absorption) == self.octave_bands.n_bands:
                # Multi-band
                abs_dict = {
                    "coeffs": self.walls[i].absorption,
                    "center_freqs": self.octave_bands.centers,
                    "description": "",
                }
                sca_dict = {
                    "coeffs": self.walls[i].scatter,
                    "center_freqs": self.octave_bands.centers,
                    "description": "",
                }
                wall_materials[name] = Material(
                    energy_absorption=abs_dict,
                    scattering=sca_dict,
                )
            else:
                raise ValueError(
                    "Encountered a material with inconsistent number of bands."
                )

        ############################
        # BEGIN COMPATIBILITY CODE #
        ############################
        if absorption is not None:
            absorption = 0.0
            absorption_compatibility_request = True
        else:
            absorption_compatibility_request = False
        ##########################
        # END COMPATIBILITY CODE #
        ##########################

        if materials is not None:
            if absorption_compatibility_request:
                import warnings

                warnings.warn(
                    "Because materials were specified, "
                    "deprecated absorption parameter is ignored.",
                    DeprecationWarning,
                )

            if not isinstance(materials, dict):
                materials = {"floor": materials, "ceiling": materials}

            for mat in materials.values():
                assert isinstance(
                    mat, Material
                ), "Material not specified using correct class"

        elif absorption_compatibility_request:
            import warnings

            warnings.warn(
                "absorption parameter is deprecated for Room.extrude",
                DeprecationWarning,
            )

            absorption = np.array(absorption)
            if absorption.ndim == 0:
                absorption = absorption * np.ones(2)
            elif absorption.ndim == 1 and absorption.shape[0] != 2:
                raise ValueError(
                    "The size of the absorption array must be 2 for extrude, "
                    "for the floor and ceiling"
                )

            materials = make_materials(
                floor=(absorption[0], 0.0), ceiling=(absorption[0], 0.0)
            )

        else:
            # In this case, no material is provided, use totally reflective walls, no scattering
            new_mat = Material(0.0, 0.0)
            materials = {"floor": new_mat, "ceiling": new_mat}

        new_corners = {}
        new_corners["floor"] = np.pad(floor_corners, ((0, 1), (0, 0)), mode="constant")
        new_corners["ceiling"] = (new_corners["floor"].T + height * v_vec).T

        # we need the floor corners to ordered clockwise (for the normal to point outward)
        new_corners["floor"] = np.fliplr(new_corners["floor"])

        # Concatenate new walls param with old ones.
        wall_corners.update(new_corners)
        wall_materials.update(materials)

        # If some of the materials used are multi-band, we need to resample
        # all of them to have the same number of values
        if not Material.all_flat(wall_materials):
            for name, mat in wall_materials.items():
                mat.resample(self.octave_bands)

        walls = []
        for key, corners in wall_corners.items():
            walls.append(
                wall_factory(
                    corners,
                    wall_materials[key].absorption_coeffs,
                    wall_materials[key].scattering_coeffs,
                    name=key,
                )
            )

        self.walls = walls
        self.dim = 3

        # Update the real room object
        self._init_room_engine()

    def plot(
        self,
        img_order=None,
        freq=None,
        figsize=None,
        no_axis=False,
        mic_marker_size=10,
        plot_directivity=True,
        ax=None,
        **kwargs,
    ):
        """Plots the room with its walls, microphones, sources and images"""

        try:
            import matplotlib
            import matplotlib.pyplot as plt
            from matplotlib.collections import PatchCollection
            from matplotlib.patches import Circle, Polygon, Wedge
        except ImportError:
            import warnings

            warnings.warn("Matplotlib is required for plotting")
            return

        fig = None

        if self.dim == 2:
            fig = plt.figure(figsize=figsize)

            if no_axis is True:
                if ax is None:
                    ax = fig.add_axes([0, 0, 1, 1], aspect="equal", **kwargs)
                ax.axis("off")
                rect = fig.patch
                rect.set_facecolor("gray")
                rect.set_alpha(0.15)
            else:
                if ax is None:
                    ax = fig.add_subplot(111, aspect="equal", **kwargs)

            # draw room
            corners = np.array([wall.corners[:, 0] for wall in self.walls]).T
            polygons = [Polygon(xy=corners.T, closed=True)]
            p = PatchCollection(
                polygons,
                cmap=matplotlib.cm.jet,
                facecolor=np.array([1, 1, 1]),
                edgecolor=np.array([0, 0, 0]),
            )
            ax.add_collection(p)

            if self.mic_array is not None:
                for i in range(self.mic_array.nmic):
                    ax.scatter(
                        self.mic_array.R[0][i],
                        self.mic_array.R[1][i],
                        marker="x",
                        linewidth=0.5,
                        s=mic_marker_size,
                        c="k",
                    )

                    if plot_directivity and self.mic_array.directivity[i] is not None:
                        azimuth_plot = np.linspace(
                            start=0, stop=360, num=361, endpoint=True
                        )
                        ax = self.mic_array.directivity[i].plot_response(
                            azimuth=azimuth_plot,
                            degrees=True,
                            ax=ax,
                            offset=self.mic_array.R[:, i],
                        )

                # draw the beam pattern of the beamformer if requested (and available)
                if (
                    freq is not None
                    and isinstance(self.mic_array, bf.Beamformer)
                    and (
                        self.mic_array.weights is not None
                        or self.mic_array.filters is not None
                    )
                ):
                    freq = np.array(freq)
                    if freq.ndim == 0:
                        freq = np.array([freq])

                    # define a new set of colors for the beam patterns
                    newmap = plt.get_cmap("autumn")
                    desat = 0.7
                    try:
                        # this is for matplotlib >= 2.0.0
                        ax.set_prop_cycle(
                            color=[
                                newmap(k) for k in desat * np.linspace(0, 1, len(freq))
                            ]
                        )
                    except:
                        # keep this for backward compatibility
                        ax.set_color_cycle(
                            [newmap(k) for k in desat * np.linspace(0, 1, len(freq))]
                        )

                    phis = np.arange(360) * 2 * np.pi / 360.0
                    newfreq = np.zeros(freq.shape)
                    H = np.zeros((len(freq), len(phis)), dtype=complex)
                    for i, f in enumerate(freq):
                        newfreq[i], H[i] = self.mic_array.response(phis, f)

                    # normalize max amplitude to one
                    H = np.abs(H) ** 2 / np.abs(H).max() ** 2

                    # a normalization factor according to room size
                    norm = np.linalg.norm(
                        (corners - self.mic_array.center), axis=0
                    ).max()

                    # plot all the beam patterns
                    for f, h in zip(newfreq, H):
                        x = np.cos(phis) * h * norm + self.mic_array.center[0, 0]
                        y = np.sin(phis) * h * norm + self.mic_array.center[1, 0]
                        ax.plot(x, y, "-", linewidth=0.5)

            # define some markers for different sources and colormap for damping
            markers = ["o", "s", "v", "."]
            cmap = plt.get_cmap("YlGnBu")

            # use this to check some image sources were drawn
            has_drawn_img = False

            # draw the scatter of images
            for i, source in enumerate(self.sources):
                # draw source
                ax.scatter(
                    source.position[0],
                    source.position[1],
                    c=[cmap(1.0)],
                    s=20,
                    marker=markers[i % len(markers)],
                    edgecolor=cmap(1.0),
                )

                if plot_directivity and source.directivity is not None:
                    azimuth_plot = np.linspace(
                        start=0, stop=360, num=361, endpoint=True
                    )
                    ax = source.directivity.plot_response(
                        azimuth=azimuth_plot,
                        degrees=True,
                        ax=ax,
                        offset=source.position,
                    )

                # draw images
                if img_order is None:
                    img_order = 0
                elif img_order == "max":
                    img_order = self.max_order

                I = source.orders <= img_order
                if len(I) > 0:
                    has_drawn_img = True

                val = (np.log2(np.mean(source.damping, axis=0)[I]) + 10.0) / 10.0
                # plot the images
                ax.scatter(
                    source.images[0, I],
                    source.images[1, I],
                    c=cmap(val),
                    s=20,
                    marker=markers[i % len(markers)],
                    edgecolor=cmap(val),
                )

            # When no image source has been drawn, we need to use the bounding box
            # to set correctly the limits of the plot
            if not has_drawn_img or img_order == 0:
                bbox = self.get_bbox()
                ax.set_xlim(bbox[0, :])
                ax.set_ylim(bbox[1, :])

            return fig, ax

        if self.dim == 3:
            import matplotlib.colors as colors
            import matplotlib.pyplot as plt
            import mpl_toolkits.mplot3d as a3

            if ax is None:
                fig = plt.figure(figsize=figsize)
                ax = a3.Axes3D(fig, auto_add_to_figure=False)
                fig.add_axes(ax)

            # plot the walls
            for w in self.walls:
                tri = a3.art3d.Poly3DCollection([w.corners.T], alpha=0.5)
                tri.set_color(colors.rgb2hex(np.random.rand(3)))
                tri.set_edgecolor("k")
                ax.add_collection3d(tri)

            # define some markers for different sources and colormap for damping
            markers = ["o", "s", "v", "."]
            cmap = plt.get_cmap("YlGnBu")

            # use this to check some image sources were drawn
            has_drawn_img = False

            # draw the scatter of images
            for i, source in enumerate(self.sources):
                # draw source
                ax.scatter(
                    source.position[0],
                    source.position[1],
                    source.position[2],
                    c=[cmap(1.0)],
                    s=20,
                    marker=markers[i % len(markers)],
                    edgecolor=cmap(1.0),
                )

                if plot_directivity and source.directivity is not None:
                    azimuth_plot = np.linspace(
                        start=0, stop=360, num=361, endpoint=True
                    )
                    colatitude_plot = np.linspace(
                        start=0, stop=180, num=180, endpoint=True
                    )
                    ax = source.directivity.plot_response(
                        azimuth=azimuth_plot,
                        colatitude=colatitude_plot,
                        degrees=True,
                        ax=ax,
                        offset=source.position,
                    )

                # draw images
                if img_order is None:
                    img_order = self.max_order

                I = source.orders <= img_order
                if len(I) > 0:
                    has_drawn_img = True

                val = (np.log2(np.mean(source.damping, axis=0)[I]) + 10.0) / 10.0
                # plot the images
                ax.scatter(
                    source.images[0, I],
                    source.images[1, I],
                    source.images[2, I],
                    c=cmap(val),
                    s=20,
                    marker=markers[i % len(markers)],
                    edgecolor=cmap(val),
                )

            # When no image source has been drawn, we need to use the bounding box
            # to set correctly the limits of the plot
            if not has_drawn_img or img_order == 0:
                bbox = self.get_bbox()
                ax.set_xlim3d(bbox[0, :])
                ax.set_ylim3d(bbox[1, :])
                ax.set_zlim3d(bbox[2, :])

            # draw the microphones
            if self.mic_array is not None:
                for i in range(self.mic_array.nmic):
                    ax.scatter(
                        self.mic_array.R[0][i],
                        self.mic_array.R[1][i],
                        self.mic_array.R[2][i],
                        marker="x",
                        linewidth=0.5,
                        s=mic_marker_size,
                        c="k",
                    )

                    if plot_directivity and self.mic_array.directivity[i] is not None:
                        azimuth_plot = np.linspace(
                            start=0, stop=360, num=361, endpoint=True
                        )
                        colatitude_plot = np.linspace(
                            start=0, stop=180, num=180, endpoint=True
                        )
                        ax = self.mic_array.directivity[i].plot_response(
                            azimuth=azimuth_plot,
                            colatitude=colatitude_plot,
                            degrees=True,
                            ax=ax,
                            offset=self.mic_array.R[:, i],
                        )

            return fig, ax

    def plot_rir(self, select=None, FD=False, kind=None):
        """
        Plot room impulse responses. Compute if not done already.

        Parameters
        ----------
        select: list of tuples OR int
            List of RIR pairs `(mic, src)` to plot, e.g. `[(0,0), (0,1)]`. Or
            `int` to plot RIR from particular microphone to all sources. Note
            that microphones and sources are zero-indexed. Default is to plot
            all microphone-source pairs.
        FD: bool, optional
            If True, the transfer function is plotted instead of the impulse response.
            Default is False.
        kind: str, optional
            The value can be "ir", "tf", or "spec" which will plot impulse response,
            transfer function, and spectrogram, respectively. If this option is
            specified, then the value of ``FD`` is ignored. Default is "ir".


        Returns
        -------
        fig: matplotlib figure
            Figure object for further modifications
        axes: matplotlib list of axes objects
            Axes for further modifications
        """

        if kind is None:
            kind = "tf" if FD else "ir"

        if kind == "ir":
            y_label = None
            x_label = "Time (ms)"
        elif kind == "tf":
            x_label = "Freq. (kHz)"
            y_label = "Power (dB)"
        elif kind == "spec":
            x_label = "Time (ms)"
            y_label = "Freq. (kHz)"
        else:
            raise ValueError("The value of 'kind' should be 'ir', 'tf', or 'spec'.")

        n_src = len(self.sources)
        n_mic = self.mic_array.M
        if select is None:
            pairs = [(r, s) for r in range(n_mic) for s in range(n_src)]
        elif isinstance(select, int):
            pairs = [(select, s) for s in range(n_src)]
        elif isinstance(select, list) or isinstance(select, tuple):
            if (
                len(select) == 2
                and isinstance(select[0], int)
                and isinstance(select[1], int)
            ):
                pairs = [select]
            else:
                pairs = select
        else:
            raise ValueError('Invalid type for "select".')

        if not self.simulator_state["rir_done"]:
            self.compute_rir()

        # for plotting
        n_mic = len(list(set(pair[0] for pair in pairs)))
        n_src = len(list(set(pair[1] for pair in pairs)))
        r_plot = dict()
        s_plot = dict()
        for k, r in enumerate(list(set(pair[0] for pair in pairs))):
            r_plot[r] = k
        for k, s in enumerate(list(set(pair[1] for pair in pairs))):
            s_plot[s] = k

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            import warnings

            warnings.warn("Matplotlib is required for plotting")
            return

        def plot_func(ax, h):
            if kind == "ir":
                ax.plot(np.arange(len(h)) / float(self.fs / 1000), h)
            elif kind == "tf":
                H = 20.0 * np.log10(abs(np.fft.rfft(h)) + 1e-15)
                freq = np.arange(H.shape[0]) / h.shape[0] * (self.fs / 1000)
                ax.plot(freq, H)
            elif kind == "spec":
                h = h + np.random.randn(*h.shape) * 1e-15
                ax.specgram(h, Fs=self.fs / 1000)
            else:
                raise ValueError("The value of 'kind' should be 'ir', 'tf', or 'spec'.")

        if select is None:
            fig, axes = plt.subplots(
                n_mic, n_src, squeeze=False, sharex=True, sharey=True
            )
            for r in range(n_mic):
                for s in range(n_src):
                    h = self.rir[r][s]
                    plot_func(axes[r, s], h)

            for r in range(n_mic):
                if y_label is not None:
                    axes[r, 0].set_ylabel(y_label)

                axes[r, -1].annotate(
                    "Mic {}".format(r),
                    xy=(1.02, 0.5),
                    xycoords="axes fraction",
                    rotation=270,
                    ha="left",
                    va="center",
                )

            for s in range(n_src):
                axes[0, s].set_title("Source {}".format(s), fontsize="medium")
                if x_label is not None:
                    axes[-1, s].set_xlabel(x_label)

            fig.align_ylabels(axes[:, 0])
            fig.tight_layout()

        else:
            fig, axes = plt.subplots(
                len(pairs), 1, squeeze=False, sharex=True, sharey=True
            )
            for k, (r, s) in enumerate(pairs):
                h = self.rir[r][s]
                plot_func(axes[k, 0], h)

                if len(pairs) == 1:
                    axes[k, 0].set_title("Mic {}, Source {}".format(r, s))
                else:
                    axes[k, 0].annotate(
                        "M{}, S{}".format(r, s),
                        xy=(1.02, 0.5),
                        xycoords="axes fraction",
                        rotation=270,
                        ha="left",
                        va="center",
                    )

                if y_label is not None:
                    axes[k, 0].set_ylabel(y_label)

            if x_label is not None:
                axes[-1, 0].set_xlabel(x_label)
            fig.align_ylabels(axes[:, 0])
            fig.tight_layout()

        return fig, axes

    def add(self, obj):
        """
        Adds a sound source or microphone to a room

        Parameters
        ----------
        obj: :py:obj:`~pyroomacoustics.soundsource.SoundSource` or :py:obj:`~pyroomacoustics.beamforming.Microphone` object
            The object to add

        Returns
        -------
        :py:obj:`~pyroomacoustics.room.Room`
            The room is returned for further tweaking.
        """

        if isinstance(obj, SoundSource):
            if obj.dim != self.dim:
                raise ValueError(
                    (
                        "The Room and SoundSource objects must be of the same "
                        "dimensionality. The Room is {}D but the SoundSource "
                        "is {}D"
                    ).format(self.dim, obj.dim)
                )

            if not self.is_inside(np.array(obj.position)):
                raise ValueError("The source must be added inside the room.")

            self.sources.append(obj)

        elif isinstance(obj, MicrophoneArray):
            if obj.dim != self.dim:
                raise ValueError(
                    (
                        "The Room and MicrophoneArray objects must be of the same "
                        "dimensionality. The Room is {}D but the MicrophoneArray "
                        "is {}D"
                    ).format(self.dim, obj.dim)
                )

            if not hasattr(self, "mic_array") or self.mic_array is None:
                self.mic_array = obj
            else:
                self.mic_array.append(obj)

            # microphone need to be added to the room_engine
            for m in range(len(obj)):
                self.room_engine.add_mic(obj.R[:, None, m])

        else:
            raise TypeError(
                "The add method from Room only takes SoundSource or "
                "MicrophoneArray objects as parameter"
            )

        return self

    def add_microphone(self, loc, fs=None, directivity=None):
        """
        Adds a single microphone in the room.

        Parameters
        ----------
        loc: array_like or ndarray
            The location of the microphone. The length should be the same as the room dimension.
        fs: float, optional
            The sampling frequency of the microphone, if different from that of the room.

        Returns
        -------
        :py:obj:`~pyroomacoustics.room.Room`
            The room is returned for further tweaking.
        """

        if self.simulator_state["rt_needed"] and directivity is not None:
            raise NotImplementedError("Directivity not supported with ray tracing.")

        if self.dim != 3 and directivity is not None:
            raise NotImplementedError("Directivity is only supported for 3D rooms.")

        # make sure this is a
        loc = np.array(loc)

        # if array, make it a 2D array as expected
        if loc.ndim == 1:
            loc = loc[:, None]

        if fs is None:
            fs = self.fs

        return self.add(MicrophoneArray(loc, fs, directivity))

    def add_microphone_array(self, mic_array, directivity=None):
        """
        Adds a microphone array (i.e. several microphones) in the room.

        Parameters
        ----------
        mic_array: array_like or ndarray or MicrophoneArray object
            The array can be provided as an array of size ``(dim, n_mics)``,
            where ``dim`` is the dimension of the room and ``n_mics`` is the
            number of microphones in the array.

            As an alternative, a
            :py:obj:`~pyroomacoustics.beamforming.MicrophoneArray` can be
            provided.
        directivity: list of Directivity objects, optional
            If ``mic_array`` is provided as a numpy array, an optional
            :py:obj:`~pyroomacoustics.directivities.Directivity` object or
            list thereof can be provided.
            If ``mic_array`` is a MicrophoneArray object, passing an argument here
            will result in an error.

        Returns
        -------
        :py:obj:`~pyroomacoustics.room.Room`
            The room is returned for further tweaking.
        """

        if self.simulator_state["rt_needed"] and directivity is not None:
            raise NotImplementedError("Directivity not supported with ray tracing.")

        if self.dim != 3 and directivity is not None:
            raise NotImplementedError("Directivity is only supported for 3D rooms.")

        if not isinstance(mic_array, MicrophoneArray):
            # if the type is not a microphone array, try to parse a numpy array
            mic_array = MicrophoneArray(mic_array, self.fs, directivity)
        else:
            # if the type is microphone array
            if directivity is not None:
                raise ValueError(
                    "When providing a MicrophoneArray object, the directivities should "
                    "be provided in the object, not via the `directivity` parameter "
                    "of this method."
                )

            if self.simulator_state["rt_needed"] and mic_array.is_directive:
                raise NotImplementedError("Directivity not supported with ray tracing.")

        return self.add(mic_array)

    def add_source(self, position, signal=None, delay=0, directivity=None):
        """
        Adds a sound source given by its position in the room. Optionally
        a source signal and a delay can be provided.

        Parameters
        -----------
        position: ndarray, shape: (2,) or (3,)
            The location of the source in the room
        signal: ndarray, shape: (n_samples,), optional
            The signal played by the source
        delay: float, optional
            A time delay until the source signal starts
            in the simulation

        Returns
        -------
        :py:obj:`~pyroomacoustics.room.Room`
            The room is returned for further tweaking.
        """

        if self.simulator_state["rt_needed"] and directivity is not None:
            raise NotImplementedError("Directivity not supported with ray tracing.")

        if self.dim != 3 and directivity is not None:
            raise NotImplementedError("Directivity is only supported for 3D rooms.")

        if directivity is not None:
            from pyroomacoustics import ShoeBox

            if not isinstance(self, ShoeBox):
                raise NotImplementedError(
                    "Source directivity only supported for ShoeBox room."
                )

        if isinstance(position, SoundSource):
            if directivity is not None:
                if isinstance(directivity, CardioidFamily) or isinstance(
                    directivity, MeasuredDirectivity
                ):
                    return self.add(SoundSource(position, directivity=directivity))
            else:
                return self.add(position)
        else:
            if directivity is not None:
                if isinstance(directivity, CardioidFamily) or isinstance(
                    directivity, MeasuredDirectivity
                ):
                    return self.add(
                        SoundSource(
                            position,
                            signal=signal,
                            delay=delay,
                            directivity=directivity,
                        )
                    )

            else:
                return self.add(SoundSource(position, signal=signal, delay=delay))

    def add_soundsource(self, sndsrc, directivity=None):
        """
        Adds a :py:obj:`pyroomacoustics.soundsource.SoundSource` object to the room.

        Parameters
        ----------
        sndsrc: :py:obj:`~pyroomacoustics.soundsource.SoundSource` object
            The SoundSource object to add to the room
        """
        if directivity is not None:
            sndsrc.set_directivity(directivity)
        return self.add(sndsrc)

    def image_source_model(self):
        if not self.simulator_state["ism_needed"]:
            return

        self.visibility = []

        for source in self.sources:
            n_visible_sources = self.room_engine.image_source_model(source.position)

            if n_visible_sources > 0:
                # Copy to python managed memory

                source.images = (
                    self.room_engine.sources.copy()
                )  # Positions of the image source (3,n) n: n_sources
                source.orders = (
                    self.room_engine.orders.copy()
                )  # Reflection order for each image source shape n:n_sources
                source.orders_xyz = (
                    self.room_engine.orders_xyz.copy()
                )  # Reflection order for each image source for each coordinate shape (3,n) n:n_sources
                source.walls = (
                    self.room_engine.gen_walls.copy()
                )  # Something that i don't get [-1,-1,-1,-1,-1...] shape n:n_sources
                source.damping = (
                    self.room_engine.attenuations.copy()
                )  # Octave band damping's shape (no_of_octave_bands*n_sources) damping value for each image source for each octave bands
                source.generators = -np.ones(source.walls.shape)

                # if randomized image method is selected, add a small random
                # displacement to the image sources

                if self.simulator_state["random_ism_needed"]:
                    n_images = np.shape(source.images)[1]

                    # maximum allowed displacement is 8cm
                    max_disp = self.max_rand_disp

                    # add a random displacement to each cartesian coordinate
                    disp = np.random.uniform(
                        -max_disp, max_disp, size=(self.dim, n_images)
                    )
                    source.images += disp

                self.visibility.append(self.room_engine.visible_mics.copy())

                # We need to check that microphones are indeed in the room
                for m in range(self.mic_array.R.shape[1]):
                    # if not, it's not visible from anywhere!
                    if not self.is_inside(self.mic_array.R[:, m]):
                        self.visibility[-1][m, :] = 0
            else:
                # if we are here, this means even the direct path is not visible
                # we set the visibility of the direct path as 0
                self.visibility.append(np.zeros((self.mic_array.M, 1), dtype=np.int32))

        # Update the state
        self.simulator_state["ism_done"] = True

    def ray_tracing(self):
        if not self.simulator_state["rt_needed"]:
            return

        # this will be a list of lists with
        # shape (n_mics, n_src, n_directions, n_bands, n_time_bins)
        self.rt_histograms = [[] for r in range(self.mic_array.M)]

        for s, src in enumerate(self.sources):
            self.room_engine.ray_tracing(self.rt_args["n_rays"], src.position)

            for r in range(self.mic_array.M):
                self.rt_histograms[r].append([])
                for h in self.room_engine.microphones[r].histograms:
                    # get a copy of the histogram
                    self.rt_histograms[r][s].append(h.get_hist())
            # reset all the receivers' histograms
            self.room_engine.reset_mics()

        # Basically, histograms for 2 mics corresponding to each source , the histograms are in each octave bands hence (7,2500) 2500 histogram length
        # update the state
        self.simulator_state["rt_done"] = True

    def compute_rir(self):
        """
        Compute the room impulse response between every source and microphone.
        """

        if self.simulator_state["ism_needed"] and not self.simulator_state["ism_done"]:
            self.image_source_model()

        if self.simulator_state["rt_needed"] and not self.simulator_state["rt_done"]:
            self.ray_tracing()

        self.rir = []

        volume_room = self.get_volume()

        # Loop over ever microphone present in the room and then for each
        # microphone and source pair present in the room
        for m, mic in enumerate(self.mic_array.R.T):
            self.rir.append([])
            for s, src in enumerate(self.sources):
                """
                Compute the room impulse response between the source
                and the microphone whose position is given as an
                argument.
                """
                # fractional delay length
                fdl = constants.get("frac_delay_length")

                rir_parts = []

                if self.simulator_state["ism_needed"]:
                    ir_ism = compute_ism_rir(
                        src,
                        mic,
                        self.mic_array.directivity[m],
                        self.visibility[s][m, :],
                        fdl,
                        self.c,
                        self.fs,
                        self.octave_bands,
                        air_abs_coeffs=self.air_absorption,
                        min_phase=self.min_phase,
                    )
                    rir_parts.append(ir_ism)

                if self.simulator_state["rt_needed"]:
                    ir_rt = compute_rt_rir(
                        self.rt_histograms[m][s],
                        self.rt_args["hist_bin_size"],
                        self.rt_args["hist_bin_size_samples"],
                        volume_room,
                        fdl,
                        self.c,
                        self.fs,
                        self.octave_bands,
                        air_abs_coeffs=self.air_absorption,
                    )
                    rir_parts.append(ir_rt)

                if len(rir_parts) == 0:
                    raise ValueError("Both ISM and RT are disabled")
                elif len(rir_parts) == 1:
                    rir = rir_parts[0]
                else:
                    max_len = max([r.shape[0] for r in rir_parts])
                    rir = np.zeros(max_len)
                    for r in rir_parts:
                        rir[: r.shape[0]] += r

                self.rir[m].append(rir)

        self.simulator_state["rir_done"] = True

    def dft_scale_rir_calc(
        self,
        attenuations,
        dist,
        time,
        bws,
        N,
        azi_m,
        col_m,
        azi_s,
        col_s,
        src_pos=0,
        mic_pos=0,
    ):
        """
        Full DFT scale RIR construction.

        This function also takes into account the FIR's of the source and receiver retrieved from the SOFA file.



        Parameters
        ----------
        attenuations: arr
            Dampings for all the image sources Shape : ( No_of_octave_band x no_img_src)
        dist : arr
            distance of all the image source present in the room from this particular mic Shape : (no_img_src)
        time : arr
            Time of arrival of all the image source Shape : (no_img_src)
        bws :
            bandwidth of all the octave bands
        N :
        azi_m : arr
            Azimuth angle of arrival of this particular mic for all image sources Shape : (no_img_src)
        col_m : arr
            Colatitude angle of arrival of this particular mic  for all image sources Shape : (no_img_src)
        azi_s : arr
            Azimuth angle of departure of this particular source for all image sources Shape : (no_img_src)
        col_s : arr
            Colatitude angle of departure of this particular source for all image sources Shape : (no_img_src)
        src_pos : int
            The particular source we are calculating RIR
        mic_pos : int
            The particular mic we are calculating RIR

        Returns
        -------
            rir : :py:class:`~numpy.ndarray`
                Constructed RIR for this particlar src mic pair .

            The constructed RIR still lacks air absorption and distance absorption because in the old pyroom these calculation happens on the octave band level.


        """

        attenuations = attenuations / dist
        alp = []
        window_length = 81

        no_imag_src = attenuations.shape[1]

        fp_im = N
        fir_length_octave_band = self.octave_bands.n_fft

        from .build_rir import (
            fast_convolution_3,
            fast_convolution_4,
            fast_window_sinc_interpolator,
        )

        rec_presence = True if (len(azi_m) > 0 and len(col_m) > 0) else False
        source_presence = True if (len(azi_s) > 0 and len(col_s) > 0) else False

        final_fir_IS_len = (
            (self.mic_array.directivity[mic_pos].filter_len_ir if (rec_presence) else 1)
            + (
                self.sources[src_pos].directivity.filter_len_ir
                if (source_presence)
                else 1
            )
            + window_length
            + fir_length_octave_band
        ) - 3

        if rec_presence and source_presence:
            resp_mic = self.mic_array.directivity[mic_pos].get_response(
                azimuth=azi_m, colatitude=col_m, degrees=False
            )  # Return response as an array of number of (img_sources * length of filters)
            resp_src = self.sources[src_pos].directivity.get_response(
                azimuth=azi_s, colatitude=col_s, degrees=False
            )

            if self.mic_array.directivity[mic_pos].filter_len_ir == 1:
                resp_mic = np.array(resp_mic).reshape(-1, 1)

            else:
                assert (
                    self.fs == self.mic_array.directivity[mic_pos].fs
                ), "Mic directivity: frequency of simulation should be same as frequency of interpolation"

            if self.sources[src_pos].directivity.filter_len_ir == 1:
                resp_src = np.array(resp_src).reshape(-1, 1)
            else:
                assert (
                    self.fs == self.sources[src_pos].directivity.fs
                ), "Source directivity:  frequency of simulation should be same as frequency of interpolation"

        else:
            if source_presence:
                assert (
                    self.fs == self.sources[src_pos].directivity.fs
                ), "Directivity source frequency of simulation should be same as frequency of interpolation"

                resp_src = self.sources[src_pos].directivity.get_response(
                    azimuth=azi_s,
                    colatitude=col_s,
                    degrees=False,
                )

            elif rec_presence:
                assert (
                    self.fs == self.mic_array.directivity[mic_pos].fs
                ), "Directivity mic frequency of simulation should be same as frequency of interpolation"

                resp_mic = self.mic_array.directivity[mic_pos].get_response(
                    azimuth=azi_m,
                    colatitude=col_m,
                    degrees=False,
                )

        # else:
        # txt = "No"
        # final_fir_IS_len = (fir_length_octave_band + window_length) - 1

        time_arrival_is = time  # For min phase

        # Calculating fraction delay sinc filter
        sample_frac = time_arrival_is * self.fs  # Find the fractional sample number

        ir_diff = np.zeros(N + (final_fir_IS_len))  # 2050 #600

        # Create arrays for fractional delay low pass filter, sum of {damping coeffiecients * octave band filter}, source response, receiver response.

        cpy_ir_len_1 = np.zeros((no_imag_src, final_fir_IS_len), dtype=np.complex_)
        cpy_ir_len_2 = np.zeros((no_imag_src, final_fir_IS_len), dtype=np.complex_)
        cpy_ir_len_3 = np.zeros((no_imag_src, final_fir_IS_len), dtype=np.complex_)
        cpy_ir_len_4 = np.zeros((no_imag_src, final_fir_IS_len), dtype=np.complex_)
        att_in_dft_scale = np.zeros(
            (no_imag_src, fir_length_octave_band), dtype=np.complex_
        )

        # Vectorized sinc filters

        vectorized_interpolated_sinc = np.zeros(
            (no_imag_src, window_length), dtype=np.double
        )
        vectorized_time_ip = np.array(
            [int(math.floor(sample_frac[img_src])) for img_src in range(no_imag_src)]
        )
        vectorized_time_fp = [
            sample_frac[img_src] - int(math.floor(sample_frac[img_src]))
            for img_src in range(no_imag_src)
        ]
        vectorized_time_fp = np.array(vectorized_time_fp, dtype=np.double)
        vectorized_interpolated_sinc = fast_window_sinc_interpolator(
            vectorized_time_fp, window_length, vectorized_interpolated_sinc
        )

        for i in range(no_imag_src):  # Loop through Image source
            att_in_octave_band = attenuations[:, i]
            att_in_dft_scale_ = att_in_dft_scale[i, :]

            # Interpolating attenuations given in the single octave band to a DFT scale.

            att_in_dft_scale_ = self.octave_bands.octave_band_dft_interpolation(
                att_in_octave_band,
                self.air_absorption,
                dist[i],
                att_in_dft_scale_,
                bws,
                self.min_phase,
            )

            # time_ip = int(math.floor(sample_frac[i]))  # Calculating the integer sample

            # time_fp = sample_frac[i] - time_ip  # Calculating the fractional sample

            # windowed_sinc_filter = fast_window_sinc_interpolater(time_fp)

            cpy_ir_len_1[i, : att_in_dft_scale_.shape[0]] = np.fft.ifft(
                att_in_dft_scale_
            )
            cpy_ir_len_2[i, :window_length] = vectorized_interpolated_sinc[i, :]

            if source_presence and rec_presence:
                cpy_ir_len_3[i, : resp_src[i, :].shape[0]] = resp_src[i, :]

                cpy_ir_len_4[i, : resp_mic[i, :].shape[0]] = resp_mic[i, :]

                out = fast_convolution_4(
                    cpy_ir_len_1[i, :],
                    cpy_ir_len_2[i, :],
                    cpy_ir_len_3[i, :],
                    cpy_ir_len_4[i, :],
                    final_fir_IS_len,
                )

                ir_diff[
                    vectorized_time_ip[i] : (vectorized_time_ip[i] + final_fir_IS_len)
                ] += np.real(out)

            else:
                if source_presence:
                    resp = resp_src[i, :]
                elif rec_presence:
                    resp = resp_mic[i, :]

                cpy_ir_len_3[i, : resp.shape[0]] = resp

                out = fast_convolution_3(
                    cpy_ir_len_1[i, :],
                    cpy_ir_len_2[i, :],
                    cpy_ir_len_3[i, :],
                    final_fir_IS_len,
                )

                ir_diff[
                    vectorized_time_ip[i] : (vectorized_time_ip[i] + final_fir_IS_len)
                ] += np.real(out)

        return ir_diff

    def simulate(
        self,
        snr=None,
        reference_mic=0,
        callback_mix=None,
        callback_mix_kwargs={},
        return_premix=False,
        recompute_rir=False,
    ):
        r"""
        Simulates the microphone signal at every microphone in the array

        Parameters
        ----------
        reference_mic: int, optional
            The index of the reference microphone to use for SNR computations.
            The default reference microphone is the first one (index 0)
        snr: float, optional
            The target signal-to-noise ratio (SNR) in decibels at the reference microphone.
            When this option is used the argument
            :py:attr:`pyroomacoustics.room.Room.sigma2_awgn` is ignored. The variance of
            every source at the reference microphone is normalized to one and
            the variance of the noise \\(\\sigma_n^2\\) is chosen

            .. math::

                \mathsf{SNR} = 10 \log_{10} \frac{ K }{ \sigma_n^2 }

            The value of :py:attr:`pyroomacoustics.room.Room.sigma2_awgn` is also set
            to \\(\\sigma_n^2\\) automatically

        callback_mix: func, optional
            A function that will perform the mix, it takes as first argument
            an array of shape ``(n_sources, n_mics, n_samples)`` that contains
            the source signals convolved with the room impulse response prior
            to mixture at the microphone. It should return an array of shape
            ``(n_mics, n_samples)`` containing the mixed microphone signals.
            If such a function is provided, the ``snr`` option is ignored
            and :py:attr:`pyroomacoustics.room.Room.sigma2_awgn` is set to ``None``.
        callback_mix_kwargs: dict, optional
            A dictionary that contains optional arguments for ``callback_mix``
            function
        return_premix: bool, optional
            If set to ``True``, the function will return an array of shape
            ``(n_sources, n_mics, n_samples)`` containing the microphone
            signals with individual sources, convolved with the room impulse
            response but prior to mixing
        recompute_rir: bool, optional
            If set to ``True``, the room impulse responses will be recomputed
            prior to simulation

        Returns
        -------
        Nothing or an array of shape ``(n_sources, n_mics, n_samples)``
            Depends on the value of ``return_premix`` option
        """

        # import convolution routine
        from scipy.signal import fftconvolve

        # Throw an error if we are missing some hardware in the room
        if len(self.sources) == 0:
            raise ValueError("There are no sound sources in the room.")
        if self.mic_array is None:
            raise ValueError("There is no microphone in the room.")

        # compute RIR if necessary
        if self.rir is None or len(self.rir) == 0 or recompute_rir:
            self.compute_rir()

        # number of mics and sources
        M = self.mic_array.M
        S = len(self.sources)

        # compute the maximum signal length
        from itertools import product

        max_len_rir = np.array(
            [len(self.rir[i][j]) for i, j in product(range(M), range(S))]
        ).max()
        f = lambda i: len(self.sources[i].signal) + np.floor(
            self.sources[i].delay * self.fs
        )
        max_sig_len = np.array([f(i) for i in range(S)]).max()
        L = int(max_len_rir) + int(max_sig_len) - 1
        if L % 2 == 1:
            L += 1

        # the array that will receive all the signals
        premix_signals = np.zeros((S, M, L))

        # compute the signal at every microphone in the array
        for m in np.arange(M):
            for s in np.arange(S):
                sig = self.sources[s].signal
                if sig is None:
                    continue
                d = int(np.floor(self.sources[s].delay * self.fs))
                h = self.rir[m][s]
                premix_signals[s, m, d : d + len(sig) + len(h) - 1] += fftconvolve(
                    h, sig
                )

        if callback_mix is not None:
            # Execute user provided callback
            signals = callback_mix(premix_signals, **callback_mix_kwargs)
            self.sigma2_awgn = None

        elif snr is not None:
            # Normalize all signals so that
            denom = np.std(premix_signals[:, reference_mic, :], axis=1)
            premix_signals /= denom[:, None, None]
            signals = np.sum(premix_signals, axis=0)

            # Compute the variance of the microphone noise
            self.sigma2_awgn = 10 ** (-snr / 10) * S

        else:
            signals = np.sum(premix_signals, axis=0)

        # add white gaussian noise if necessary
        if self.sigma2_awgn is not None:
            signals += np.random.normal(0.0, np.sqrt(self.sigma2_awgn), signals.shape)

        # record the signals in the microphones
        self.mic_array.record(signals, self.fs)

        if return_premix:
            return premix_signals

    def direct_snr(self, x, source=0):
        """Computes the direct Signal-to-Noise Ratio"""

        if source >= len(self.sources):
            raise ValueError("No such source")

        if self.sources[source].signal is None:
            raise ValueError("No signal defined for source " + str(source))

        if self.sigma2_awgn is None:
            return float("inf")

        x = np.array(x)
        sigma2_s = np.mean(self.sources[0].signal ** 2)
        d2 = np.sum((x - self.sources[source].position) ** 2)

        return sigma2_s / self.sigma2_awgn / (16 * np.pi**2 * d2)

    def get_wall_by_name(self, name):
        """
        Returns the instance of the wall by giving its name.

        Parameters
        ----------
        name: string
            name of the wall

        Returns
        -------
        Wall
            instance of the wall with this name
        """

        if name in self.wallsId:
            return self.walls[self.wallsId[name]]
        else:
            raise ValueError("The wall " + name + " cannot be found.")

    def get_bbox(self):
        """Returns a bounding box for the room"""

        lower = np.amin(np.concatenate([w.corners for w in self.walls], axis=1), axis=1)
        upper = np.amax(np.concatenate([w.corners for w in self.walls], axis=1), axis=1)

        return np.c_[lower, upper]

    def is_inside(self, p, include_borders=True):
        """
        Checks if the given point is inside the room.

        Parameters
        ----------
        p: array_like, length 2 or 3
            point to be tested
        include_borders: bool, optional
            set true if a point on the wall must be considered inside the room

        Returns
        -------
            True if the given point is inside the room, False otherwise.
        """

        p = np.array(p)
        if self.dim != p.shape[0]:
            raise ValueError("Dimension of room and p must match.")

        # The method works as follows: we pick a reference point *outside* the room and
        # draw a line between the point to check and the reference.
        # If the point to check is inside the room, the line will intersect an odd
        # number of walls. If it is outside, an even number.
        # Unfortunately, there are a lot of corner cases when the line intersects
        # precisely on a corner of the room for example, or is aligned with a wall.

        # To avoid all these corner cases, we will do a randomized test.
        # We will pick a point at random outside the room so that the probability
        # a corner case happen is virtually zero. If the test raises a corner
        # case, we will repeat the test with a different reference point.

        # get the bounding box
        bbox = self.get_bbox()
        bbox_center = np.mean(bbox, axis=1)
        bbox_max_dist = np.linalg.norm(bbox[:, 1] - bbox[:, 0]) / 2

        # re-run until we get a non-ambiguous result
        it = 0
        while it < constants.get("room_isinside_max_iter"):
            # Get random point outside the bounding box
            random_vec = np.random.randn(self.dim)
            random_vec /= np.linalg.norm(random_vec)
            p0 = bbox_center + 2 * bbox_max_dist * random_vec

            ambiguous = False  # be optimistic
            is_on_border = False  # we have to know if the point is on the boundary
            count = 0  # wall intersection counter
            for i in range(len(self.walls)):
                # intersects, border_of_wall, border_of_segment = self.walls[i].intersects(p0, p)
                # ret = self.walls[i].intersects(p0, p)
                loc = np.zeros(self.dim, dtype=np.float32)
                ret = self.walls[i].intersection(p0, p, loc)

                if (
                    ret == int(Wall.Isect.ENDPT) or ret == 3
                ):  # this flag is True when p is on the wall
                    is_on_border = True

                elif ret == Wall.Isect.BNDRY:
                    # the intersection is on a corner of the room
                    # but the point to check itself is *not* on the wall
                    # then things get tricky
                    ambiguous = True

                # count the wall intersections
                if ret >= 0:  # valid intersection
                    count += 1

            # start over when ambiguous
            if ambiguous:
                it += 1
                continue

            else:
                if is_on_border and not include_borders:
                    return False
                elif is_on_border and include_borders:
                    return True
                elif count % 2 == 1:
                    return True
                else:
                    return False

        return False

        # We should never reach this
        raise ValueError(
            """
                Error could not determine if point is in or out in maximum number of iterations.
                This is most likely a bug, please report it.
                """
        )

    def wall_area(self, wall):
        """Computes the area of a 3D planar wall.

        Parameters
        ----------
        wall: Wall instance
            the wall object that is defined in 3D space

        """

        # Algo : http://geomalgorithms.com/a01-_area.

        # Recall that the wall corners have the following shape :
        # [  [x1, x2, ...], [y1, y2, ...], [z1, z2, ...]  ]

        c = wall.corners
        n = wall.normal / np.linalg.norm(wall.normal)

        if len(c) != 3:
            raise ValueError("The function wall_area3D only supports ")

        sum_vect = [0.0, 0.0, 0.0]
        num_vertices = len(c[0])

        for i in range(num_vertices):
            sum_vect = sum_vect + np.cross(c[:, (i - 1) % num_vertices], c[:, i])

        return abs(np.dot(n, sum_vect)) / 2.0

    def get_volume(self):
        """
        Computes the volume of the room

        Returns
        -------
        float
            the volume of the room
        """
        wall_sum = 0.0

        for w in self.walls:
            n = (w.normal) / np.linalg.norm(w.normal)
            one_point = w.corners[:, 0]

            wall_sum += np.dot(n, one_point) * w.area()

        return wall_sum / 3.0

    @property
    def volume(self):
        return self.get_volume()

    @property
    def n_mics(self):
        return len(self.mic_array) if self.mic_array is not None else 0

    @property
    def n_sources(self):
        return len(self.sources) if self.sources is not None else 0

    def rt60_theory(self, formula="sabine"):
        """
        Compute the theoretical reverberation time (RT60) for the room.

        Parameters
        ----------
        formula: str
            The formula to use for the calculation, 'sabine' (default) or 'eyring'
        """

        rt60 = 0.0

        if self.is_multi_band:
            bandwidths = self.octave_bands.get_bw()
        else:
            bandwidths = [1.0]

        V = self.volume
        S = np.sum([w.area() for w in self.walls])
        c = self.c

        for i, bw in enumerate(bandwidths):
            # average absorption coefficients
            a = 0.0
            for w in self.walls:
                if len(w.absorption) == 1:
                    a += w.area() * w.absorption[0]
                else:
                    a += w.area() * w.absorption[i]
            a /= S

            try:
                m = self.air_absorption[i]
            except:
                m = 0.0

            if formula == "eyring":
                rt60_loc = rt60_eyring(S, V, a, m, c)
            elif formula == "sabine":
                rt60_loc = rt60_sabine(S, V, a, m, c)
            else:
                raise ValueError("Only Eyring and Sabine's formulas are supported")

            rt60 += rt60_loc * bw

        rt60 /= np.sum(bandwidths)
        return rt60

    def measure_rt60(self, decay_db=60, plot=False):
        """
        Measures the reverberation time (RT60) of the simulated RIR.

        Parameters
        ----------
        decay_db: float
            This is the actual decay of the RIR used for the computation. The
            default is 60, meaning that the RT60 is exactly what we measure.
            In some cases, the signal may be too short  to measure 60 dB decay.
            In this case, we can specify a lower value. For example, with 30
            dB, the RT60 is twice the time measured.
        plot: bool
            Displays a graph of the Schroeder curve and the estimated RT60.

        Returns
        -------
        ndarray (n_mics, n_sources)
            An array that contains the measured RT60 for all the RIR.
        """

        rt60 = np.zeros((self.n_mics, self.n_sources))

        for m in range(self.n_mics):
            for s in range(self.n_sources):
                rt60[m, s] = measure_rt60(
                    self.rir[m][s], fs=self.fs, plot=plot, decay_db=decay_db
                )

        return rt60


class ShoeBox(Room):
    """
    This class provides an API for creating a ShoeBox room in 2D or 3D.

    Parameters
    ----------
    p : array_like
        Length 2 (width, length) or 3 (width, length, height) depending on
        the desired dimension of the room.
    fs: int, optional
        The sampling frequency in Hz. Default is 8000.
    t0: float, optional
        The global starting time of the simulation in seconds. Default is 0.
    absorption : float
        Average amplitude absorption of walls. Note that this parameter is
        deprecated; use `materials` instead!
    max_order: int, optional
        The maximum reflection order in the image source model. Default is 1,
        namely direct sound and first order reflections.
    sigma2_awgn: float, optional
        The variance of the additive white Gaussian noise added during
        simulation. By default, none is added.
    sources: list of SoundSource objects, optional
        Sources to place in the room. Sources can be added after room creating
        with the `add_source` method by providing coordinates.
    mics: MicrophoneArray object, optional
        The microphone array to place in the room. A single microphone or
        microphone array can be added after room creation with the
        `add_microphone_array` method.
    materials : `Material` object or `dict` of `Material` objects
        See `pyroomacoustics.parameters.Material`. If providing a `dict`,
        you must provide a `Material` object for each wall: 'east',
        'west', 'north', 'south', 'ceiling' (3D), 'floor' (3D).
    temperature: float, optional
        The air temperature in the room in degree Celsius. By default, set so
        that speed of sound is 343 m/s.
    humidity: float, optional
        The relative humidity of the air in the room (between 0 and 100). By
        default set to 0.
    air_absorption: bool, optional
        If set to True, absorption of sound energy by the air will be
        simulated.
    ray_tracing: bool, optional
        If set to True, the ray tracing simulator will be used along with
        image source model.
    use_rand_ism: bool, optional
        If set to True, image source positions will have a small random
        displacement to prevent sweeping echoes
    max_rand_disp: float, optional;
        If using randomized image source method, what is the maximum
        displacement of the image sources?
    """

    def __init__(
        self,
        p,
        fs=8000,
        t0=0.0,
        absorption=None,  # deprecated
        max_order=1,
        sigma2_awgn=None,
        sources=None,
        mics=None,
        materials=None,
        temperature=None,
        humidity=None,
        air_absorption=False,
        ray_tracing=False,
        use_rand_ism=False,
        max_rand_disp=0.08,
        min_phase=False,
    ):
        p = np.array(p, dtype=np.float32)

        if len(p.shape) > 1 and (len(p) != 2 or len(p) != 3):
            raise ValueError("`p` must be a vector of length 2 or 3.")

        self.dim = p.shape[0]

        # record shoebox dimension in object
        self.shoebox_dim = np.array(p)

        # initialize the attributes of the room
        self._var_init(
            fs,
            t0,
            max_order,
            sigma2_awgn,
            temperature,
            humidity,
            air_absorption,
            ray_tracing,
            use_rand_ism,
            max_rand_disp,
            min_phase,
        )

        # Keep the correctly ordered naming of walls
        # This is the correct order for the shoebox computation later
        # W/E is for axis x, S/N for y-axis, F/C for z-axis
        self.wall_names = ["west", "east", "south", "north"]
        if self.dim == 3:
            self.wall_names += ["floor", "ceiling"]

        n_walls = len(self.wall_names)

        ############################
        # BEGIN COMPATIBILITY CODE #
        ############################

        if absorption is None:
            absorption_compatibility_request = False
            absorption = 0.0
        else:
            absorption_compatibility_request = True

        # copy over the absorption coefficient
        if isinstance(absorption, float):
            absorption = dict(zip(self.wall_names, [absorption] * n_walls))

        ##########################
        # END COMPATIBILITY CODE #
        ##########################

        if materials is not None:
            if absorption_compatibility_request:
                warnings.warn(
                    "Because `materials` were specified, deprecated "
                    "`absorption` parameter is ignored.",
                    DeprecationWarning,
                )

            if isinstance(materials, Material):
                materials = dict(zip(self.wall_names, [materials] * n_walls))
            elif not isinstance(materials, dict):
                raise ValueError(
                    "`materials` must be a `Material` object or "
                    "a `dict` specifying a `Material` object for"
                    " each wall: 'east', 'west', 'north', "
                    "'south', 'ceiling' (3D), 'floor' (3D)."
                )

            for w_name in self.wall_names:
                assert isinstance(
                    materials[w_name], Material
                ), "Material not specified using correct class"

        elif absorption_compatibility_request:
            warnings.warn(
                "Using absorption parameter is deprecated. Use `materials` with "
                "`Material` object instead.",
                DeprecationWarning,
            )

            # order the wall absorptions
            if not isinstance(absorption, dict):
                raise ValueError(
                    "`absorption` must be either a scalar or a "
                    "2x dim dictionary with entries for each "
                    "wall, namely: 'east', 'west', 'north', "
                    "'south', 'ceiling' (3d), 'floor' (3d)."
                )

            materials = {}
            for w_name in self.wall_names:
                if w_name in absorption:
                    # Fix the absorption
                    # 1 - a1 == sqrt(1 - a2)    <-- a1 is former incorrect absorption, a2 is the correct definition based on energy
                    # <=> a2 == 1 - (1 - a1) ** 2
                    correct_abs = 1.0 - (1.0 - absorption[w_name]) ** 2
                    materials[w_name] = Material(energy_absorption=correct_abs)
                else:
                    raise KeyError(
                        "Absorption needs to have keys 'east', 'west', "
                        "'north', 'south', 'ceiling' (3d), 'floor' (3d)."
                    )
        else:
            # In this case, no material is provided, use totally reflective
            # walls, no scattering
            materials = dict(
                zip(self.wall_names, [Material(energy_absorption=0.0)] * n_walls)
            )

        # If some of the materials used are multi-band, we need to resample
        # all of them to have the same number of values
        if not Material.all_flat(materials):
            for name, mat in materials.items():
                mat.resample(self.octave_bands)

        # Get the absorption and scattering as arrays
        # shape: (n_bands, n_walls)
        absorption_array = np.array(
            [materials[w].absorption_coeffs for w in self.wall_names]
        ).T
        scattering_array = np.array(
            [materials[w].scattering_coeffs for w in self.wall_names]
        ).T

        # Create the real room object
        self._init_room_engine(self.shoebox_dim, absorption_array, scattering_array)

        self.walls = self.room_engine.walls

        Room._wall_mapping(self)

        # add the sources
        self.sources = []
        if sources is not None and isinstance(sources, list):
            for src in sources:
                self.add_soundsource(src)

        # add the microphone array
        if mics is not None:
            self.add_microphone_array(mics)
        else:
            self.mic_array = None

    def extrude(self, height):
        """Overload the extrude method from 3D rooms"""

        if height < 0.0:
            raise ValueError("Room height must be positive")

        Room.extrude(self, np.array([0.0, 0.0, height]))

        # update the shoebox dim
        self.shoebox_dim = np.append(self.shoebox_dim, height)

    def get_volume(self):
        """
        Computes the volume of a room

        Returns
        -------
        the volume in cubic unit
        """

        return np.prod(self.shoebox_dim)

    def is_inside(self, pos):
        """
        Parameters
        ----------
        pos: array_like
            The position to test in an array of size 2 for a 2D room and 3 for a 3D room

        Returns
        -------
        True if ``pos`` is a point in the room, ``False`` otherwise.
        """
        pos = np.array(pos)
        return np.all(pos >= 0) and np.all(pos <= self.shoebox_dim)


class AnechoicRoom(ShoeBox):
    """
    This class provides an API for creating an Anechoic "room" in 2D or 3D.

    Parameters
    ----------
    dim: int
        Dimension of the room (2 or 3).
    fs: int, optional
        The sampling frequency in Hz. Default is 8000.
    t0: float, optional
        The global starting time of the simulation in seconds. Default is 0.
    sigma2_awgn: float, optional
        The variance of the additive white Gaussian noise added during
        simulation. By default, none is added.
    sources: list of SoundSource objects, optional
        Sources to place in the room. Sources can be added after room creating
        with the `add_source` method by providing coordinates.
    mics: MicrophoneArray object, optional
        The microphone array to place in the room. A single microphone or
        microphone array can be added after room creation with the
        `add_microphone_array` method.
    temperature: float, optional
        The air temperature in the room in degree Celsius. By default, set so
        that speed of sound is 343 m/s.
    humidity: float, optional
        The relative humidity of the air in the room (between 0 and 100). By
        default set to 0.
    air_absorption: bool, optional
        If set to True, absorption of sound energy by the air will be
        simulated.
    """

    def __init__(
        self,
        dim=3,
        fs=8000,
        t0=0.0,
        sigma2_awgn=None,
        sources=None,
        mics=None,
        temperature=None,
        humidity=None,
        air_absorption=False,
    ):
        if not dim in [2, 3]:
            raise ValueError("Anechoic room dimension has to be either 2 or 3.")

        # Setting max_order to 0 emulates an anechoic room.
        max_order = 0

        # Ray tracing only makes sense in echoic rooms.
        ray_tracing = False

        # Create some dummy walls
        p = np.ones((dim,))

        # The materials are not actually used because max_order is set to 0 and ray-tracing to False.
        # Anyways, we use the energy_absorption and scattering corresponding to an anechoic room.
        materials = Material(energy_absorption=1.0, scattering=0.0)

        # Set deprecated parameter
        absorption = None

        ShoeBox.__init__(
            self,
            p=p,
            fs=fs,
            t0=t0,
            max_order=max_order,
            sigma2_awgn=sigma2_awgn,
            sources=sources,
            mics=mics,
            materials=materials,
            temperature=temperature,
            humidity=humidity,
            air_absorption=air_absorption,
            ray_tracing=ray_tracing,
        )

    def __str__(self):
        return "AnechoicRoom instance in {}D.".format(self.dim)

    def is_inside(self, p):
        """Overloaded function to eliminate testing if objects are inside "room"."""
        # always return True because we want the walls to have no effect.
        return True

    def get_bbox(self):
        """Returns a bounding box for the mics and sources, for plotting."""

        if (self.mic_array is None) and not self.sources:
            raise ValueError("Nothing to plot, the Anechoic Room is empty!")

        lower = np.inf * np.ones((self.dim,))
        upper = -np.inf * np.ones((self.dim,))

        if self.mic_array is not None:
            lower = np.min(np.column_stack((lower, self.mic_array.R)), axis=1)
            upper = np.max(np.column_stack((upper, self.mic_array.R)), axis=1)

        for i, source in enumerate(self.sources):
            lower = np.min(np.row_stack((lower, source.position)), axis=0)
            upper = np.max(np.row_stack((upper, source.position)), axis=0)

        return np.c_[lower, upper]

    def plot_walls(self, ax):
        """Overloaded function to eliminate wall plotting."""
        return 1

    def plot(self, **kwargs):
        """Overloaded function to issue warning when img_order is given."""
        if "img_order" in kwargs.keys():
            warnings.warn("Ignoring img_order argument for AnechoicRoom.", UserWarning)
        ShoeBox.plot(self, **kwargs)
