# Main Room class using to encapsulate the room acoustics simulator
# Copyright (C) 2023-2014  Robin Scheibler, Ivan Dokmanic, Sidney Barthe, Cyril Cadoux
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
from .anechoic import AnechoicRoom
from .helpers import find_non_convex_walls, sequence_generation, wall_factory
from .room import Room
from .shoebox import ShoeBox
