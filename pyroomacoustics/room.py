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

r'''
Room
====

The three main classes are :py:obj:`pyroomacoustics.room.Room`,
:py:obj:`pyroomacoustics.soundsource.SoundSource`, and
:py:obj:`pyroomacoustics.beamforming.MicrophoneArray`. On a high level, a
simulation scenario is created by first defining a room to which a few sound
sources and a microphone array are attached. The actual audio is attached to
the source as raw audio samples. The image source method (ISM) is then used to
find all image sources up to a maximum specified order and room impulse
responses (RIR) are generated from their positions. The microphone signals are
then created by convolving the audio samples associated to sources with the
appropriate RIR. Since the simulation is done on discrete-time signals, a
sampling frequency is specified for the room and the sources it contains.
Microphones can optionally operate at a different sampling frequency; a rate
conversion is done in this case.

Simulating a Shoebox Room
-------------------------

We will first walk through the steps to simulate a shoebox-shaped room in 3D.


Create the room
~~~~~~~~~~~~~~~

So-called shoebox rooms are pallelepipedic rooms with 4 or 6 walls (in 2D and 3D,
respectiely), all at right angles. They are defined by a single vector that contains
the lengths of the walls. They have the advantage of being simple to define and very
efficient to simulate. A ``9m x 7.5m x 3.5m`` room is simply defined like this

.. code-block:: python

    import pyroomacoustics as pra
    room = pra.ShoeBox([9, 7.5, 3.5], fs=16000, absorption=0.35, max_order=17)

The second argument is the sampling frequency at which the RIR will be
generated. Note that the default value of ``fs`` is 8 kHz. The third argument
is the absorption of the walls, namely reflections are multiplied by ``(1 -
absorption)`` for every wall they hit. The fourth argument is the maximum
number of reflections allowed in the ISM.

The relationship between ``absorption``/``max_order`` and `reverberation time
<https://en.wikipedia.org/wiki/Reverberation>`_ (the T60 or RT60 in the
acoustics literature) is not straightforward. `Sabine's formula
<https://en.wikipedia.org/wiki/Reverberation#Sabine_equation>`_ can be used to
some extent to set these parameters.


Add sources and microphones
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sources are fairly straighforward to create. They take their location as single
mandatory argument, and a signal and start time as optional arguments.  Here we
create a source located at ``[2.5, 3.73, 1.76]`` within the room, that will utter
the content of the wav file ``speech.wav`` starting at ``1.3 s`` into the simulation.

.. code-block:: python

    # import a mono wavfile as the source signal
    # the sampling frequency should match that of the room
    from scipy.io import wavfile
    _, audio = wavfile.read('speech.wav')

    my_source = pra.SoundSource([2.5, 3.73, 1.76], signal=audio, delay=1.3)

    # place the source in the room
    room.add_source(my_source)

The locations of the microphones in the array should be provided in a numpy
``nd-array`` of size ``(ndim, nmics)``, that is each column contains the
coordinates of one microphone. This array is used to construct a
:py:obj:`pyroomacoustics.beamforming.MicrophoneArray` object, together with the
sampling frequency for the microphone. Note that it can be different from that
of the room, in which case resampling will occur. Here, we create an array
with two microphones placed at ``[6.3, 4.87, 1.2]`` and ``[6.3, 4.93, 1.2]``.

.. code-block:: python

    # define the location of the array
    import numpy as np
    R = np.c_[
        [6.3, 4.87, 1.2],  # mic 1
        [6.3, 4.93, 1.2],  # mic 2
        ]

    # the fs of the microphones is the same as the room
    mic_array = pra.MicrophoneArray(R, room.fs)

    # finally place the array in the room
    room.add_microphone_array(mic_array)

A number of routines exist to create regular array geometries in 2D.

- :py:func:`pyroomacoustics.beamforming.linear_2D_array`
- :py:func:`pyroomacoustics.beamforming.circular_2D_array`
- :py:func:`pyroomacoustics.beamforming.square_2D_array`
- :py:func:`pyroomacoustics.beamforming.poisson_2D_array`
- :py:func:`pyroomacoustics.beamforming.spiral_2D_array`


Create the Room Impulse Response
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At this point, the RIRs are simply created by invoking the ISM via
:py:func:`pyroomacoustics.room.Room.image_source_model`. This function will
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


Simulate sound propagation
~~~~~~~~~~~~~~~~~~~~~~~~~~

By calling :py:func:`pyroomacoustics.room.Room.simulate`, a convolution of the
signal of each source (if not ``None``) will be performed with the
corresponding room impulse response. The output from the convolutions will be summed up
at the microphones. The result is stored in the ``signals`` attribute of ``room.mic_array``
with each row corresponding to one microphone.

.. code-block:: python

    room.simulate()

    # plot signal at microphone 1
    plt.plot(room.mic_array.signals[1,:])

Controlling the signal-to-noise ratio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


Example
-------

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import pyroomacoustics as pra

    # Create a 4 by 6 metres shoe box room
    room = pra.ShoeBox([4,6])

    # Add a source somewhere in the room
    room.add_source([2.5, 4.5])

    # Create a linear array beamformer with 4 microphones
    # with angle 0 degrees and inter mic distance 10 cm
    R = pra.linear_2D_array([2, 1.5], 4, 0, 0.04)
    room.add_microphone_array(pra.Beamformer(R, room.fs))

    # Now compute the delay and sum weights for the beamformer
    room.mic_array.rake_delay_and_sum_weights(room.sources[0][:1])

    # plot the room and resulting beamformer
    room.plot(freq=[1000, 2000, 4000, 8000], img_order=0)
    plt.show()

'''


from __future__ import print_function

import math
import numpy as np
import scipy.spatial as spatial
from scipy.signal import sosfiltfilt

#import .beamforming as bf
from . import beamforming as bf
from .soundsource import SoundSource
from .acoustics import OctaveBandsFactory
from .parameters import constants, eps, Physics, Material
from .utilities import fractional_delay

from . import libroom
from .libroom import Wall, Wall2D


def wall_factory(corners, absorption, scattering, name=""):
    ''' Call the correct method according to wall dimension '''
    if corners.shape[0] == 3:
        return Wall(corners, absorption, 0., name)
    elif corners.shape[0] == 2:
        return Wall2D(corners, absorption, 0., name)
    else:
        raise ValueError('Rooms can only be 2D or 3D')


class Room(object):
    '''
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

    The Room is sub-classed by :py:obj:pyroomacoustics.room.ShoeBox` which
    creates a rectangular (2D) or parallelepipedic (3D) room. Such rooms
    benefit from an efficient algorithm for the image source method.


    :attribute walls: (Wall array) list of walls forming the room
    :attribute fs: (int) sampling frequency
    :attribute t0: (float) time offset
    :attribute max_order: (int) the maximum computed order for images
    :attribute sources: (SoundSource array) list of sound sources
    :attribute mics: (MicrophoneArray) array of microphones
    :attribute corners: (numpy.ndarray 2xN or 3xN, N=number of walls) array containing a point belonging to each wall, used for calculations
    :attribute absorption: (numpy.ndarray size N, N=number of walls)  array containing the absorption factor for each wall, used for calculations
    :attribute dim: (int) dimension of the room (2 or 3 meaning 2D or 3D)
    :attribute wallsId: (int dictionary) stores the mapping "wall name -> wall id (in the array walls)"
    '''

    def __init__(
            self,
            walls,
            fs=8000,
            t0=0.,
            temperature=25.,
            humidity=70.,
            c=None,
            air_absorption=None,
            max_order=1,
            ray_trace_args=None,
            sources=None,
            mics=None,
            ):

        self.walls = walls

        # Get the room dimension from that of the walls
        self.dim = walls[0].dim

        # initialize everything else
        self._var_init(fs, t0, temperature, humidity, c, air_absorption, max_order, sources, mics)

        self._wall_mapping()

        # check which walls are part of the convex hull
        self.convex_hull()

        # process arguments for ray tracing
        self._ray_trace_args_init(ray_trace_args)

        args = [
                self.walls,
                self.obstructing_walls,
                [],
                self.air_absorption,
                self.c,  # speed of sound
                self.max_order,
                self.rt_args['energy_threshold'],
                self.rt_args['time_threshold'],
                self.rt_args['receiver_radius'],
                self.rt_args['hist_bin_size'],
                True,  # a priori we will always use a hybrid model
                ]

        # Create the real room object
        if self.dim == 2:
            self.room_engine = libroom.Room2(*args)
        else:
            self.room_engine = libroom.Room(*args)



    def _ray_trace_args_init(self, rt_args):

        rt_args_default = {
                'n_rays' : 10000,
                'energy_thres' : 1e-7,
                'time_thres' : 1.,
                'receiver_radius' : 0.15,
                'hist_bin_size' : 0.004,
                }

        if rt_args is None:
            self.rt_args = {}
        else:
            self.rt_args = rt_args.copy()

        for key, val in rt_args_default.items():
            if key not in self.rt_args:
                self.rt_args[key] = val


    def _var_init(self, fs, t0, temperature, humidity, c, air_absorption, max_order, sources, mics):

        self.fs = fs
        self.octave_bands = OctaveBandsFactory(fs=self.fs)

        self.max_order = max_order

        self.physics = Physics(temperature=temperature, humidity=humidity)
        if c is None:
            self.c = self.physics.get_sound_speed()
        else:
            # ignore temperature and humidity if c is provided
            self.c = c

        if air_absorption is None:
            self.air_absorption = self.octave_bands(**self.physics.get_air_absorption())
        else:
            # ignore temperature and humidity if coefficients are provided directly
            self.air_absorption = self.octave_bands(**air_absorption)

        # Compute the filter delay if not provided
        if t0 < (constants.get('frac_delay_length')-1)/float(fs)/2:
            self.t0 = (constants.get('frac_delay_length')-1)/float(fs)/2
        else:
            self.t0 = t0

        if sources is not None and isinstance(sources, list):
            self.sources = sources
        else:
            self.sources = []

        self.mic_array = mics

        # in the beginning, nothing has been
        self.visibility = None

        # initialize the attribute for the impulse responses
        self.rir = None


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
            materials=None,
            fs=8000,
            **kwargs,
            ):
        '''
        Creates a 2D room by giving an array of corners.

        :arg corners: (np.array dim 2xN, N>2) list of corners, must be antiClockwise oriented
        :arg absorption: (float array or float) list of absorption factor for each wall or single value for all walls

        :returns: (Room) instance of a 2D room
        '''
        n_walls = corners.shape[1]

        corners = np.array(corners)
        if (corners.shape[0] != 2 or n_walls < 3):
            raise ValueError('Arg corners must be more than two 2D points.')

        # We want to make sure the corners are ordered counter-clockwise
        if (libroom.area_2d_polygon(corners) <= 0):
            corners = corners[:,::-1]

        ############################
        # BEGIN COMPATIBILITY CODE #
        ############################

        if absorption is None:
            absorption = 0.
            absorption_compatibility_request = False
        else:
            absorption_compatibility_request = True

        absorption = np.array(absorption, dtype='float64')
        if (absorption.ndim == 0):
            absorption = absorption * np.ones(n_walls)
        elif (absorption.ndim >= 1 and n_walls != len(absorption)):
            raise ValueError(
                    'Arg absorption must be the same size as corners or must be a single value.'
                    )

        ############################
        # BEGIN COMPATIBILITY CODE #
        ############################

        if materials is not None:

            if absorption_compatibility_request:
                import warnings
                warnings.warn(
                        'Because materials were specified, deprecated absorption parameter is ignored.',
                        DeprecationWarning,
                        )

            if not isinstance(materials, list):
                materials = [materials] * n_walls

            if len(materials) != n_walls:
                raise ValueError(
                        'One material per wall is necessary.'
                        )

            for i in range(n_walls):
                assert isinstance(materials[i], Material), 'Material not specified using correct class'

        elif absorption_compatibility_request:
            import warnings
            warnings.warn('Using absorption parameter is deprecated. In the future, use materials instead.')

            # Fix the absorption
            # 1 - a1 == sqrt(1 - a2)    <-- a1 is former incorrect absorption, a2 is the correct definition based on energy
            # <=> a2 == 1 - (1 - a1) ** 2
            correct_absorption = 1. - (1. - absorption) ** 2
            materials = [Material.make_freq_flat(a) for a in correct_absorption]

        else:
            # In this case, no material is provided, use totally reflective walls, no scattering
            materials = [Material.make_freq_flat(0., 0.)] * n_walls

        # Resample material properties at octave bands
        octave_bands = OctaveBandsFactory(fs=fs)
        if not Material.all_flat(materials):
            for mat in materials:
                mat.resample(octave_bands)

        # Create the walls
        walls = []
        for i in range(n_walls):
            walls.append(wall_factory(
                np.array([corners[:, i], corners[:, (i+1) % n_walls]]).T,
                materials[i].get_abs(),
                materials[i].get_scat(),
                "wall_"+str(i),
                ))

        return cls(walls, fs=fs, **kwargs)

    def extrude(
            self,
            height,
            v_vec=None,
            materials=None,
            absorption=None,
            ):
        '''
        Creates a 3D room by extruding a 2D polygon.
        The polygon is typically the floor of the room and will have z-coordinate zero. The ceiling

        Parameters
        ----------
        height : float
            The extrusion height
        v_vec : array-like 1D length 3, optionnal
            A unit vector. An orientation for the extrusion direction. The
            ceiling will be placed as a translation of the floor with respect
            to this vector (The default is [0,0,1]).
        absorption : float or array-like
            Absorption coefficients for all the walls. If a scalar, then all the walls
            will have the same absorption. If an array is given, it should have as many elements
            as there will be walls, that is the number of vertices of the polygon plus two. The two
            last elements are for the floor and the ceiling, respectively. (default 1)
        '''

        if self.dim != 2:
            raise ValueError('Can only extrude a 2D room.')

        # default orientation vector is pointing up
        if v_vec is None:
            v_vec = np.array([0., 0., 1.])

        # check that the walls are ordered counterclock wise
        # that should be the case if created from from_corners function
        nw = len(self.walls)
        floor_corners = np.zeros((2,nw))
        floor_corners[:,0] = self.walls[0].corners[:,0]
        ordered = True
        for iw, wall in enumerate(self.walls[1:]):
            if not np.allclose(self.walls[iw].corners[:,1], wall.corners[:,0]):
                ordered = False
            floor_corners[:,iw+1] = wall.corners[:,0]
        if not np.allclose(self.walls[-1].corners[:,1], self.walls[0].corners[:,0]):
            ordered = False

        if not ordered:
            raise ValueError("The wall list should be ordered counter-clockwise, which is the case \
                if the room is created with Room.from_corners")

        # make sure the floor_corners are ordered anti-clockwise (for now)
        if (libroom.area_2d_polygon(floor_corners) <= 0):
            floor_corners = np.fliplr(floor_corners)

        walls = []
        for i in range(nw):
            corners = np.array([
                np.r_[floor_corners[:,i], 0],
                np.r_[floor_corners[:,(i+1)%nw], 0],
                np.r_[floor_corners[:,(i+1)%nw], 0] + height*v_vec,
                np.r_[floor_corners[:,i], 0] + height*v_vec
                ]).T
            walls.append(wall_factory(
                corners, self.walls[i].absorption, self.walls[i].scatter, name=str(i)
                ))

        ############################
        # BEGIN COMPATIBILITY CODE #
        ############################
        if absorption is not None:
            absorption = 0.
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
                assert isinstance(mat, Material), 'Material not specified using correct class'

        elif absorption_compatibility_request:

            import warnings
            warnings.warn(
                    'absorption parameter is deprecated for Room.extrude',
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

            materials = {
                    "floor": Material.make_freq_flat(absorption[0], 0.),
                    "ceiling": Material.make_freq_flat(absorption[0], 0.)
                    }

        else:
            # In this case, no material is provided, use totally reflective walls, no scattering
            new_mat = Material.make_freq_flat(0., 0.)
            materials = {"floor": new_mat, "ceiling": new_mat}

        new_corners = {}
        new_corners["floor"] = np.pad(floor_corners, ((0, 1),(0,0)), mode='constant')
        new_corners["ceiling"] = (new_corners["floor"].T + height*v_vec).T

        # we need the floor corners to ordered clockwise (for the normal to point outward)
        new_corners["floor"] = np.fliplr(new_corners["floor"])

        for key in ["floor", "ceiling"]:
            walls.append(wall_factory(
                new_corners[key],
                materials[key].get_abs(),
                materials[key].get_scat(),
                name=key,
                ))

        self.walls = walls
        self.dim = 3

        # recheck which walls are in the convex hull
        self.convex_hull()

        args = [
                self.walls,
                self.obstructing_walls,
                [],
                self.air_absorption,
                self.c,  # speed of sound
                self.max_order,
                self.rt_args['energy_threshold'],
                self.rt_args['time_threshold'],
                self.rt_args['receiver_radius'],
                self.rt_args['hist_bin_size'],
                True,  # a priori we will always use a hybrid model
                ]

        # Create the real room object
        self.room_engine = libroom.Room(*args)

    def convex_hull(self):

        '''
        Finds the walls that are not in the convex hull
        '''

        all_corners = []
        for wall in self.walls[1:]:
            all_corners.append(wall.corners.T)
        X = np.concatenate(all_corners, axis=0)
        convex_hull = spatial.ConvexHull(X, incremental=True)

        # Now we need to check which walls are on the surface
        # of the hull
        self.in_convex_hull = [False] * len(self.walls)
        for i, wall in enumerate(self.walls):
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
                        self.in_convex_hull[i] = True

                elif point.shape[0] == 3:
                    # Check if co-planar
                    p0 = convex_hull.points[simplex[0]]
                    p1 = convex_hull.points[simplex[1]]
                    p2 = convex_hull.points[simplex[2]]

                    normal = np.cross(p1 - p0, p2 - p0)
                    if np.abs(np.inner(normal, point - p0)) < eps:
                        # co-planar point found!
                        self.in_convex_hull[i] = True

        self.obstructing_walls = [i for i in range(len(self.walls)) if not self.in_convex_hull[i]]


    def plot(self, img_order=None, freq=None, figsize=None, no_axis=False, mic_marker_size=10, **kwargs):
        ''' Plots the room with its walls, microphones, sources and images '''

        try:
            import matplotlib
            from matplotlib.patches import Circle, Wedge, Polygon
            from matplotlib.collections import PatchCollection
            import matplotlib.pyplot as plt
        except ImportError:
            import warnings
            warnings.warn('Matplotlib is required for plotting')
            return

        if (self.dim == 2):
            fig = plt.figure(figsize=figsize)

            if no_axis is True:
                ax = fig.add_axes([0, 0, 1, 1], aspect='equal', **kwargs)
                ax.axis('off')
                rect = fig.patch
                rect.set_facecolor('gray')
                rect.set_alpha(0.15)
            else:
                ax = fig.add_subplot(111, aspect='equal', **kwargs)

            # draw room
            corners = np.array([wall.corners[:, 0] for wall in self.walls]).T
            polygons = [Polygon(corners.T, True)]
            p = PatchCollection(polygons, cmap=matplotlib.cm.jet,
                    facecolor=np.array([1, 1, 1]), edgecolor=np.array([0, 0, 0]))
            ax.add_collection(p)

            # draw the microphones
            if (self.mic_array is not None):
                for mic in self.mic_array.R.T:
                    ax.scatter(mic[0], mic[1],
                            marker='x', linewidth=0.5, s=mic_marker_size, c='k')

                # draw the beam pattern of the beamformer if requested (and available)
                if freq is not None \
                        and isinstance(self.mic_array, bf.Beamformer) \
                        and (self.mic_array.weights is not None or self.mic_array.filters is not None):

                    freq = np.array(freq)
                    if freq.ndim == 0:
                        freq = np.array([freq])

                    # define a new set of colors for the beam patterns
                    newmap = plt.get_cmap('autumn')
                    desat = 0.7
                    try:
                        # this is for matplotlib >= 2.0.0
                        ax.set_prop_cycle(color=[newmap(k) for k in desat*np.linspace(0, 1, len(freq))])
                    except:
                        # keep this for backward compatibility
                        ax.set_color_cycle([newmap(k) for k in desat*np.linspace(0, 1, len(freq))])

                    phis = np.arange(360) * 2 * np.pi / 360.
                    newfreq = np.zeros(freq.shape)
                    H = np.zeros((len(freq), len(phis)), dtype=complex)
                    for i, f in enumerate(freq):
                        newfreq[i], H[i] = self.mic_array.response(phis, f)

                    # normalize max amplitude to one
                    H = np.abs(H)**2/np.abs(H).max()**2

                    # a normalization factor according to room size
                    norm = np.linalg.norm((corners - self.mic_array.center), axis=0).max()

                    # plot all the beam patterns
                    i = 0
                    for f, h in zip(newfreq, H):
                        x = np.cos(phis) * h * norm + self.mic_array.center[0, 0]
                        y = np.sin(phis) * h * norm + self.mic_array.center[1, 0]
                        ax.plot(x, y, '-', linewidth=0.5)

            # define some markers for different sources and colormap for damping
            markers = ['o', 's', 'v', '.']
            cmap = plt.get_cmap('YlGnBu')
            # draw the scatter of images
            for i, source in enumerate(self.sources):
                # draw source
                ax.scatter(
                    source.position[0],
                    source.position[1],
                    c=cmap(1.),
                    s=20,
                    marker=markers[i %len(markers)],
                    edgecolor=cmap(1.))

                # draw images
                if (img_order is None):
                    img_order = self.max_order

                I = source.orders <= img_order

                val = (np.log2(np.mean(source.damping, axis=0)[I]) + 10.) / 10.
                # plot the images
                ax.scatter(source.images[0, I],
                    source.images[1, I],
                    c=cmap(val),
                    s=20,
                    marker=markers[i % len(markers)],
                    edgecolor=cmap(val))

            return fig, ax

        if(self.dim==3):

            import mpl_toolkits.mplot3d as a3
            import matplotlib.colors as colors
            import matplotlib.pyplot as plt
            import scipy as sp

            fig = plt.figure(figsize=figsize)
            ax = a3.Axes3D(fig)

            # plot the walls
            for w in self.walls:
                tri = a3.art3d.Poly3DCollection([w.corners.T], alpha=0.5)
                tri.set_color(colors.rgb2hex(sp.rand(3)))
                tri.set_edgecolor('k')
                ax.add_collection3d(tri)

            # define some markers for different sources and colormap for damping
            markers = ['o', 's', 'v', '.']
            cmap = plt.get_cmap('YlGnBu')
            # draw the scatter of images
            for i, source in enumerate(self.sources):
                # draw source
                ax.scatter(
                    source.position[0],
                    source.position[1],
                    source.position[2],
                    c=cmap(1.),
                    s=20,
                    marker=markers[i %len(markers)],
                    edgecolor=cmap(1.))

                # draw images
                if (img_order is None):
                    img_order = self.max_order

                I = source.orders <= img_order

                val = (np.log2(np.mean(source.damping, axis=0)[I]) + 10.) / 10.
                # plot the images
                ax.scatter(source.images[0, I],
                    source.images[1, I],
                    source.images[2, I],
                    c=cmap(val),
                    s=20,
                    marker=markers[i % len(markers)],
                    edgecolor=cmap(val))


            # draw the microphones
            if (self.mic_array is not None):
                for mic in self.mic_array.R.T:
                    ax.scatter(mic[0], mic[1], mic[2],
                            marker='x', linewidth=0.5, s=mic_marker_size, c='k')


            return fig, ax

    def plot_rir(self, FD=False):

        if self.rir is None:
            self.compute_rir()

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            import warnings
            warnings.warn('Matplotlib is required for plotting')
            return

        from . import utilities as u

        M = self.mic_array.M
        S = len(self.sources)

        n_max = np.max([len(self.rir[r][s]) for s in range(S) for r in range(M)])

        for r in range(M):
            for s in range(S):

                h = self.rir[r][s]

                h = np.zeros((n_max))
                h[:len(self.rir[r][s])] = self.rir[r][s]
                plt.subplot(M, S, r*S + s + 1)
                if not FD:
                    plt.plot(np.arange(len(h)) / float(self.fs), h)
                else:
                    u.real_spectrum(h)
                plt.title('RIR: mic'+str(r)+' source'+str(s))
                if r == M-1:
                    if not FD:
                        plt.xlabel('Time [s]')
                    else:
                        plt.xlabel('Normalized frequency')

        plt.tight_layout()

    def add_microphone_array(self, micArray):
        self.mic_array = micArray

        for m in range(self.mic_array.M):
            self.room_engine.add_mic(self.mic_array.R[:,m])

    def add_source(self, position, signal=None, delay=0):

        if (not self.is_inside(np.array(position))):
            raise ValueError('The source must be added inside the room.')

        self.sources.append(
                SoundSource(
                    position,
                    signal=signal,
                    delay=delay
                    )
                )

    def image_source_model(self):

        self.visibility = []

        for source in self.sources:

            n_sources = self.room_engine.image_source_model(source.position)

            if (n_sources > 0):

                # Copy to python managed memory
                source.images = self.room_engine.sources.copy()
                source.orders = self.room_engine.orders.copy()
                source.walls = self.room_engine.gen_walls.copy()
                source.damping = self.room_engine.attenuations.copy()
                source.generators = -np.ones(source.walls.shape)

                self.visibility.append(self.room_engine.visible_mics.copy())

                # We need to check that microphones are indeed in the room
                for m in range(self.mic_array.R.shape[1]):
                    # if not, it's not visible from anywhere!
                    if not self.is_inside(self.mic_array.R[:,m]):
                        self.visibility[-1][m,:] = 0


    def compute_rir(self, mode='ism'):
        ''' Compute the room impulse response between every source and microphone.
        :param mode: a string that defines which method to use to compute the RIR.
                    It can take values :
                    'ism' for pure image source method,
                    'rt' for pure ray tracing,
                    'hybrid' for a mix of both methods

        All other default params are needed for the ray tracing method'''

        self.rir = []

        for m, mic in enumerate(self.mic_array.R.T):
            self.rir.append([])
            for s, src in enumerate(self.sources):

                '''
                Compute the room impulse response between the source
                and the microphone whose position is given as an
                argument.
                '''

                # fractional delay length
                fdl = constants.get('frac_delay_length')

                # compute the distance
                dist = np.sqrt(np.sum((src.images - mic[:, None])**2, axis=0))
                time = dist / self.c + self.t0

                # the number of samples needed
                N = math.ceil((time.max() - self.t0) * self.fs) + fdl

                t = np.arange(N) / float(self.fs)
                ir = np.zeros(t.shape)

                bp_filt = self.octave_bands.get_filters() if self.multi_band else [None]
                for b, bpf in enumerate(bp_filt):

                    ir_loc = np.zeros_like(ir)

                    alpha = src.damping[b, :] / dist

                    # Try to use the Cython extension
                    from .build_rir import fast_rir_builder
                    vis = self.visibility[s][m,:].astype(np.int32)
                    fast_rir_builder(ir_loc, time, alpha, vis, self.fs, fdl)

                    if bpf is not None:
                        ir += sosfiltfilt(bpf, ir_loc)
                    else:
                        ir += ir_loc

                self.rir[-1].append(ir)

    def simulate(self,
            snr=None,
            reference_mic=0,
            callback_mix=None,
            callback_mix_kwargs={},
            return_premix=False,
            recompute_rir=False,
            ):
        r'''
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
        '''

        # import convolution routine
        from scipy.signal import fftconvolve

        # Throw an error if we are missing some hardware in the room
        if (len(self.sources) == 0):
            raise ValueError('There are no sound sources in the room.')
        if (self.mic_array is None):
            raise ValueError('There is no microphone in the room.')

        # compute RIR if necessary
        if self.rir is None or len(self.rir) == 0 or recompute_rir:
            self.compute_rir()

        # number of mics and sources
        M = self.mic_array.M
        S = len(self.sources)

        # compute the maximum signal length
        from itertools import product
        max_len_rir = np.array([len(self.rir[i][j])
                                for i, j in product(range(M), range(S))]).max()
        f = lambda i: len(
            self.sources[i].signal) + np.floor(self.sources[i].delay * self.fs)
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
                premix_signals[s,m,d:d + len(sig) + len(h) - 1] += fftconvolve(h, sig)

        if callback_mix is not None:
            # Execute user provided callback
            signals = callback_mix(premix_signals, **callback_mix_kwargs)
            self.sigma2_awgn = None

        elif snr is not None:
            # Normalize all signals so that
            denom = np.std(premix_signals[:,reference_mic,:], axis=1)
            premix_signals /= denom[:,None,None]
            signals = np.sum(premix_signals, axis=0)

            # Compute the variance of the microphone noise
            self.sigma2_awgn = 10**(- snr / 10) * S

        else:
            signals = np.sum(premix_signals, axis=0)

        # add white gaussian noise if necessary
        if self.sigma2_awgn is not None:
            signals += np.random.normal(0., np.sqrt(self.sigma2_awgn), signals.shape)

        # record the signals in the microphones
        self.mic_array.record(signals, self.fs)

        if return_premix:
            return premix_signals


    def get_wall_by_name(self, name):
        '''
        Returns the instance of the wall by giving its name.

        :arg name: (string) name of the wall

        :returns: (Wall) instance of the wall with this name
        '''

        if (name in self.wallsId):
            return self.walls[self.wallsId[name]]
        else:
            raise ValueError('The wall '+name+' cannot be found.')


    def get_bbox(self):
        ''' Returns a bounding box for the room '''

        lower = np.amin(np.concatenate([w.corners for w in self.walls], axis=1), axis=1)
        upper = np.amax(np.concatenate([w.corners for w in self.walls], axis=1), axis=1)

        return np.c_[lower, upper]


    def is_inside(self, p, include_borders = True):
        '''
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
        '''

        p = np.array(p)
        if (self.dim != p.shape[0]):
            raise ValueError('Dimension of room and p must match.')

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
        bbox_max_dist = np.linalg.norm(bbox[:,1] - bbox[:,0]) / 2

        # re-run until we get a non-ambiguous result
        max_iter = 5
        it = 0
        while it < max_iter:

            # Get random point outside the bounding box
            random_vec = np.random.randn(self.dim)
            random_vec /= np.linalg.norm(random_vec)
            p0 = bbox_center + 2 * bbox_max_dist * random_vec

            ambiguous = False  # be optimistic
            is_on_border = False  # we have to know if the point is on the boundary
            count = 0  # wall intersection counter
            for i in range(len(self.walls)):
                #intersects, border_of_wall, border_of_segment = self.walls[i].intersects(p0, p)
                #ret = self.walls[i].intersects(p0, p)
                loc = np.zeros(self.dim, dtype=np.float32)
                ret = self.walls[i].intersection(p0, p, loc)

                if ret == int(Wall.Isect.ENDPT) or ret == 3:  # this flag is True when p is on the wall
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

        # We should never reach this
        raise ValueError(
                '''
                Error could not determine if point is in or out in maximum number of iterations.
                This is most likely a bug, please report it.
                '''
                )

    def wall_area(self, wall):

        """Computes the area of a 3D planar wall.
        :param wall: the wall object that is defined in the 3D space"""

        # Algo : http://geomalgorithms.com/a01-_area.

        # Recall that the wall corners have the following shape :
        # [  [x1, x2, ...], [y1, y2, ...], [z1, z2, ...]  ]

        c = wall.corners
        n = wall.normal/np.linalg.norm(wall.normal)

        if len(c) != 3:
            raise ValueError("The function wall_area3D only supports ")

        sum_vect = [0., 0., 0.]
        num_vertices = len(c[0])

        for i in range(num_vertices):
            sum_vect = sum_vect + np.cross(c[:, (i - 1) % num_vertices], c[:, i])

        return abs(np.dot(n, sum_vect)) / 2.


    def get_volume(self):

        """
        Computes the volume of a room
        :param room: the room object
        :return: the volume in cubic unit
        """

        wall_sum = 0.

        for w in self.walls:
            n = (w.normal) / np.linalg.norm(w.normal)
            one_point = w.corners[:, 0]

            wall_sum += np.dot(n, one_point) * self.wall_area(w)

        return wall_sum / 3.


# Room 3D

class ShoeBox(Room):
    '''
    This class extends room for shoebox room in 3D space.
    '''

    def __init__(self,
            p,
            absorption=None,  # deprecated
            materials=None,
            fs=8000,
            t0=0.,
            temperature=25.,
            humidity=70.,
            c=None,
            air_absorption=None,
            max_order=1,
            ray_trace_args=None,
            sources=None,
            mics=None,
            ):

        Room._var_init(self, fs, t0, temperature, humidity, c, air_absorption, max_order, sources, mics)

        p = np.array(p, dtype=np.float32)

        if len(p.shape) > 1:
            raise ValueError("p must be a vector of length 2 or 3.")

        self.dim = p.shape[0]

        # if only one point is provided, place the other at origin
        p2 = np.array(p)
        p1 = np.zeros(self.dim)

        # record shoebox dimension in object
        self.shoebox_dim = p2

        # Keep the correctly ordered naming of walls
        # This is the correct order for the shoebox computation later
        # W/E is for axis x, S/N for y-axis, F/C for z-axis
        self.wall_names = ['west', 'east', 'south', 'north']
        if self.dim == 3:
            self.wall_names += ['floor', 'ceiling']

        n_walls = len(self.wall_names)

        ############################
        # BEGIN COMPATIBILITY CODE #
        ############################

        if absorption is None:
            absorption_compatibility_request = False
            absorption = 0.
        else:
            absorption_compatibility_request = True

        import warnings
        warnings.warn('absorption parameter is deprecated for ShoeBox', DeprecationWarning)

        # copy over the aborption coefficent
        if isinstance(absorption, float):
            absorption = dict(
                    zip(self.wall_names, [absorption] * n_walls)
                    )

        ##########################
        # END COMPATIBILITY CODE #
        ##########################

        if materials is not None:

            if absorption_compatibility_request:
                import warnings
                warnings.warn(
                        'Because materials were specified, deprecated absorption parameter is ignored.',
                        DeprecationWarning,
                        )

            if isinstance(materials, Material):
                materials = dict(zip(self.wall_names, [materials] * n_walls))
            elif not isinstance(materials, dict):
                raise ValueError('Materials must be a string, Material object, or dictionary')

            for w_name in self.wall_names:
                assert isinstance(materials[w_name], Material), 'Material not specified using correct class'

        elif absorption_compatibility_request:

            import warnings
            warnings.warn('Using absorption parameter is deprecated. Use materials instead.')

            # order the wall absorptions
            if not isinstance(absorption, dict):
                raise ValueError("Absorption must be either a scalar or a 2x dim dictionnary with entries for 'east', 'west', etc.")

            materials = {}
            for w_name in self.wall_names:
                if w_name in absorption:
                    # Fix the absorption
                    # 1 - a1 == sqrt(1 - a2)    <-- a1 is former incorrect absorption, a2 is the correct definition based on energy
                    # <=> a2 == 1 - (1 - a1) ** 2
                    correct_abs = 1. - (1. - absorption[w_name]) ** 2
                    materials[w_name] = Material.make_freq_flat(absorption=correct_abs)
                else:
                    raise KeyError(
                            "Absorption needs to have keys 'east', 'west', 'north', 'south', 'ceiling' (3d), 'floor' (3d)"
                            )
        else:

            # In this case, no material is provided, use totally reflective walls, no scattering
            materials = dict(zip(self.wall_names, [Material.make_freq_flat(absorption=0.)] * n_walls))

        # At this point, we should determine if the simulation is single or multi-band
        self.multi_band = not Material.all_flat(materials)

        if self.multi_band:
            for mat in materials.values():
                mat.resample(self.octave_bands)

        # Get the absorption and scattering as arrays
        # shape: (n_bands, n_walls)
        absorption_array = np.array(
                [ materials[w].get_abs() for w in self.wall_names ]
                ).T
        scattering_array = np.array(
                [ materials[w].get_scat() for w in self.wall_names ]
                ).T

        # process arguments for ray tracing
        Room._ray_trace_args_init(self, ray_trace_args)

        args = [
                self.shoebox_dim,
                absorption_array,
                scattering_array,
                [],
                self.air_absorption,
                self.c,  # speed of sound
                max_order,
                self.rt_args['energy_thres'],
                self.rt_args['time_thres'],
                self.rt_args['receiver_radius'],
                self.rt_args['hist_bin_size'],
                True,  # a priori we will always use a hybrid model
                ]

        # Create the real room object
        if self.dim == 2:
            self.room_engine = libroom.Room2D(*args)
        else:
            self.room_engine = libroom.Room(*args)

        self.walls = self.room_engine.walls

        Room._wall_mapping(self)


    def extrude(self, height):
        ''' Overload the extrude method from 3D rooms '''

        if height < 0.:
            raise ValueError('Room height must be positive')

        Room.extrude(self, np.array([0., 0., height]))

        # update the shoebox dim
        self.shoebox_dim = np.append(self.shoebox_dim, height)



