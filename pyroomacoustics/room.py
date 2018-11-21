# @version: 1.0  date: 05/06/2015 by Sidney Barthe
# @author: robin.scheibler@epfl.ch, ivan.dokmanic@epfl.ch, sidney.barthe@epfl.ch
# @copyright: EPFL-IC-LCAV 2015
'''
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

import numpy as np
import scipy.spatial as spatial
import ctypes

#import .beamforming as bf
from . import beamforming as bf
from .soundsource import SoundSource
from .parameters import constants, eps

from . import libroom
from .libroom import Wall

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
    into 3D space by adding vertical walls and a parallel “ceiling” (see Figure
    4b).

    The Room is sub-classed by :py:obj:pyroomacoustics.room.ShoeBox` which
    creates a rectangular (2D) or parallelepipedic (3D) room. Such rooms
    benefit from an efficient algorithm for the image source method.

    
    :attribute walls: (Wall array) list of walls forming the room
    :attribute fs: (int) sampling frequency
    :attribute t0: (float) time offset
    :attribute max_order: (int) the maximum computed order for images
    :attribute sigma2_awgn: (float) ambient additive white gaussian noise level
    :attribute sources: (SoundSource array) list of sound sources
    :attribute mics: (MicrophoneArray) array of microphones
    :attribute normals: (numpy.ndarray 2xN or 3xN, N=number of walls) array containing normal vector for each wall, used for calculations
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
            max_order=1,
            sigma2_awgn=None,
            sources=None,
            mics=None):

        self.walls = walls
        self.fs = fs
        self.max_order = max_order
        self.sigma2_awgn = sigma2_awgn

        # Compute the filter delay if not provided
        if t0 < (constants.get('frac_delay_length')-1)/float(fs)/2:
            self.t0 = (constants.get('frac_delay_length')-1)/float(fs)/2
        else:
            self.t0 = t0
        
        if (sources is list):
            self.sources = sources
        else:
            self.sources = []

        self.mic_array = mics
         
        self.corners = np.array([wall.corners[:, 0] for wall in self.walls]).T
        self.absorption = np.array([wall.absorption for wall in self.walls])

        # Pre-compute RIR if needed
        if (len(self.sources) > 0 and self.mic_array is not None):
            self.compute_rir()
        else:
            self.rir = None

        # in the beginning, nothing has been 
        self.visibility = None

        # Get the room dimension from that of the walls
        self.dim = walls[0].dim

        # mapping between wall names and indices
        self.wallsId = {}
        for i in range(len(walls)):
            if self.walls[i].name is not None:
                self.wallsId[self.walls[i].name] = i

        # check which walls are part of the convex hull
        self.convex_hull()

    @classmethod
    def from_corners(
            cls,
            corners,
            absorption=0.,
            fs=8000,
            t0=0.,
            max_order=1,
            sigma2_awgn=None,
            sources=None,
            mics=None):
        '''
        Creates a 2D room by giving an array of corners.
        
        :arg corners: (np.array dim 2xN, N>2) list of corners, must be antiClockwise oriented
        :arg absorption: (float array or float) list of absorption factor for each wall or single value for all walls
        
        :returns: (Room) instance of a 2D room
        '''
        
        corners = np.array(corners)
        if (corners.shape[0] != 2 or corners.shape[1] < 3):
            raise ValueError('Arg corners must be more than two 2D points.')

        if (libroom.area_2d_polygon(corners) <= 0):
            corners = corners[:,::-1]

        cls.corners = corners
        cls.dim = corners.shape[0] 
            
        absorption = np.array(absorption, dtype='float64')
        if (absorption.ndim == 0):
            absorption = absorption * np.ones(corners.shape[1])
        elif (absorption.ndim >= 1 and corners.shape[1] != len(absorption)):
            raise ValueError('Arg absorption must be the same size as corners or must be a single value.')
        
        walls = []
        for i in range(corners.shape[1]):
            walls.append(Wall(np.array([corners[:, i], corners[:, (i+1)%corners.shape[1]]]).T, absorption[i], "wall_"+str(i)))
            
        return cls(walls, fs, t0, max_order, sigma2_awgn, sources, mics)

    def extrude(
            self,
            height,
            v_vec=None,
            absorption=0.):
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
            walls.append(Wall(corners, self.walls[i].absorption, name=str(i)))

        absorption = np.array(absorption)
        if absorption.ndim == 0:
            absorption = absorption * np.ones(2)
        elif absorption.ndim == 1 and absorption.shape[0] != 2:
            raise ValueError("The size of the absorption array must be 2 for extrude, for the floor and ceiling")

        floor_corners = np.pad(floor_corners, ((0, 1),(0,0)), mode='constant')
        ceiling_corners = (floor_corners.T + height*v_vec).T

        # we need the floor corners to ordered clockwise (for the normal to point outward)
        floor_corners = np.fliplr(floor_corners)

        walls.append(Wall(floor_corners, absorption[0], name='floor'))
        walls.append(Wall(ceiling_corners, absorption[1], name='ceiling'))

        self.walls = walls
        self.dim = 3

        # re-collect all normals, corners, absoption
        self.corners = np.array([wall.corners[:, 0] for wall in self.walls]).T
        self.absorption = np.array([wall.absorption for wall in self.walls])

        # recheck which walls are in the convex hull
        self.convex_hull()


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
            polygons = [Polygon(self.corners.T, True)]
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
                    if freq.ndim is 0:
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
                    norm = np.linalg.norm((self.corners - self.mic_array.center), axis=0).max()

                    # plot all the beam patterns
                    i = 0
                    for f, h in zip(newfreq, H):
                        x = np.cos(phis) * h * norm + self.mic_array.center[0, 0]
                        y = np.sin(phis) * h * norm + self.mic_array.center[1, 0]
                        l = ax.plot(x, y, '-', linewidth=0.5)

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

                val = (np.log2(source.damping[I]) + 10.) / 10.
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

                val = (np.log2(source.damping[I]) + 10.) / 10.
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
        for r in range(M):
            for s in range(S):
                h = self.rir[r][s]
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

        c_room = libroom.Room(self.walls, self.obstructing_walls, self.mic_array.R)

        for source in self.sources:

            # if libroom is available, use it!

            source_position = source.position.astype(np.float32)

            if isinstance(self, ShoeBox):

                # create absorption list in correct order for shoebox algorithm
                absorption_list_shoebox = np.array(
                        [self.absorption_dict[d] for d in self.wall_names],
                        )

                # Call the dedicated C routine for shoebox room
                c_room.image_source_shoebox(
                        source.position,
                        self.shoebox_dim,
                        absorption_list_shoebox,
                        self.max_order,
                        )

            else:
                # Call the general image source generator
                c_room.image_source_model(
                        source.position,
                        self.max_order,
                        )

            # Recover all the arrays as ndarray from the c struct
            n_sources = c_room.sources.shape[1]

            if (n_sources > 0):

                # Copy to python managed memory
                source.images = c_room.sources.copy()
                source.orders = c_room.orders.copy()
                source.walls = c_room.gen_walls.copy()
                source.damping = c_room.attenuations.copy()
                source.generators = -np.ones(source.walls.shape)

                self.visibility.append(c_room.visible_mics.copy())

                # We need to check that microphones are indeed in the room
                for m in range(self.mic_array.R.shape[1]):
                    # if not, it's not visible from anywhere!
                    if not self.is_inside(self.mic_array.R[:,m]):
                        self.visibility[-1][m,:] = 0


    def compute_rir(self):
        ''' Compute the room impulse response between every source and microphone '''
        
        self.rir = []

        # Run image source model if this hasn't been done
        if self.visibility is None:
            self.image_source_model()

        for m, mic in enumerate(self.mic_array.R.T):
            h = []
            for s, source in enumerate(self.sources):
                h.append(source.get_rir(mic, self.visibility[s][m], self.fs, self.t0))
            self.rir.append(h)


    def simulate(self, recompute_rir=False):
        ''' Simulates the microphone signal at every microphone in the array '''

        # import convolution routine
        from scipy.signal import fftconvolve

        # Throw an error if we are missing some hardware in the room
        if (len(self.sources) is 0):
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
        signals = np.zeros((M, L))

        # compute the signal at every microphone in the array
        for m in np.arange(M):
            rx = signals[m]
            for s in np.arange(S):
                sig = self.sources[s].signal
                if sig is None:
                    continue
                d = int(np.floor(self.sources[s].delay * self.fs))
                h = self.rir[m][s]
                rx[d:d + len(sig) + len(h) - 1] += fftconvolve(h, sig)

            # add white gaussian noise if necessary
            if self.sigma2_awgn is not None:
                rx += np.random.normal(0., np.sqrt(self.sigma2_awgn), rx.shape)

        # record the signals in the microphones
        self.mic_array.record(signals, self.fs)


    def direct_snr(self, x, source=0):
        ''' Computes the direct Signal-to-Noise Ratio '''

        if source >= len(self.sources):
            raise ValueError('No such source')

        if self.sources[source].signal is None:
            raise ValueError('No signal defined for source ' + str(source))

        if self.sigma2_awgn is None:
            return float('inf')

        x = np.array(x)

        sigma2_s = np.mean(self.sources[0].signal**2)

        d2 = np.sum((x - self.sources[source].position)**2)

        return sigma2_s/self.sigma2_awgn/(16*np.pi**2*d2)


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


# Room 3D

class ShoeBox(Room):
    '''
    This class extends room for shoebox room in 3D space.
    '''

    def __init__(self, 
            p,
            fs=8000,
            t0=0.,
            absorption=0.,
            max_order=1,
            sigma2_awgn=None,
            sources=None,
            mics=None):

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

        # copy over the aborption coefficent
        if isinstance(absorption, float):
            self.absorption_dict = dict(zip(self.wall_names, [absorption] * len(self.wall_names)))
            absorption = self.absorption_dict

        self.absorption = []
        if isinstance(absorption, dict):
            self.absorption_dict = absorption
            for d in self.wall_names:
                if d in self.absorption_dict:
                    self.absorption.append(self.absorption_dict[d])
                else:
                    raise KeyError(
                            "Absorbtion needs to have keys 'east', 'west', 'north', 'south', 'ceiling' (3d), 'floor' (3d)"
                            )

            self.absorption = np.array(self.absorption)
        else:
            raise ValueError("Absorption must be either a scalar or a 2x dim dictionnary with entries for 'east', 'west', etc.")


        if self.dim == 2:
            walls = []
            # seems the order of walls is important here, don't change!
            walls.append(Wall(np.array([[p1[0], p2[0]], [p1[1], p1[1]]]), absorption['south'], "south"))
            walls.append(Wall(np.array([[p2[0], p2[0]], [p1[1], p2[1]]]), absorption['east'], "east"))
            walls.append(Wall(np.array([[p2[0], p1[0]], [p2[1], p2[1]]]), absorption['north'], "north"))
            walls.append(Wall(np.array([[p1[0], p1[0]], [p2[1], p1[1]]]), absorption['west'], "west"))

        elif self.dim == 3:
            walls = []
            walls.append(Wall(np.array([[p1[0], p1[0], p1[0], p1[0]], [p2[1], p1[1], p1[1], p2[1]], [p1[2], p1[2], p2[2], p2[2]]]), absorption['west'], "west"))
            walls.append(Wall(np.array([[p2[0], p2[0], p2[0], p2[0]], [p1[1], p2[1], p2[1], p1[1]], [p1[2], p1[2], p2[2], p2[2]]]), absorption['east'], "east"))
            walls.append(Wall(np.array([[p1[0], p2[0], p2[0], p1[0]], [p1[1], p1[1], p1[1], p1[1]], [p1[2], p1[2], p2[2], p2[2]]]), absorption['south'], "south"))
            walls.append(Wall(np.array([[p2[0], p1[0], p1[0], p2[0]], [p2[1], p2[1], p2[1], p2[1]], [p1[2], p1[2], p2[2], p2[2]]]), absorption['north'], "north"))
            walls.append(Wall(np.array([[p2[0], p1[0], p1[0], p2[0]], [p1[1], p1[1], p2[1], p2[1]], [p1[2], p1[2], p1[2], p1[2]]]), absorption['floor'], "floor"))
            walls.append(Wall(np.array([[p2[0], p2[0], p1[0], p1[0]], [p1[1], p2[1], p2[1], p1[1]], [p2[2], p2[2], p2[2], p2[2]]]), absorption['ceiling'], "ceiling"))

        else:
            raise ValueError("Only 2D and 3D rooms are supported.")

        Room.__init__(self, walls, fs, t0, max_order, sigma2_awgn, sources, mics)

    def extrude(self, height):
        ''' Overload the extrude method from 3D rooms '''

        Room.extrude(self, np.array([0., 0., height]))


