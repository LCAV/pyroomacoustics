# @version: 1.0  date: 05/06/2015 by Sidney Barthe
# @author: robin.scheibler@epfl.ch, ivan.dokmanic@epfl.ch, sidney.barthe@epfl.ch
# @copyright: EPFL-IC-LCAV 2015

from __future__ import print_function

import numpy as np
import scipy.spatial as spatial
import ctypes

#import .beamforming as bf
from . import beamforming as bf
from . import geometry as geom
from .soundsource import SoundSource
from .wall import Wall
from .geometry import area, ccw3p
from .parameters import constants, eps

from .c_package import libroom_available, CWALL, CROOM, libroom, c_wall_p, c_int_p, c_float_p, c_room_p

class Room(object):
    '''
    This class represents a room instance.
    
    A room instance is formed by wall instances. A MicrophoneArray and SoundSources can be added.
    
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
         
        self.normals = np.array([wall.normal for wall in self.walls]).T
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

        if (geom.area(corners) <= 0):
            corners = corners[:,::-1]

        cls.corners = corners
        cls.dim = corners.shape[0] 
            
        absorption = np.array(absorption, dtype='float64')
        if (absorption.ndim == 0):
            absorption = absorption * np.ones(corners.shape[1])
        elif (absorption.ndim > 1 and corners.shape[1] != len(absorption)):
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
        if (geom.area(floor_corners) <= 0):
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
        self.normals = np.array([wall.normal for wall in self.walls]).T
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
                    if geom.ccw3p(p0, p1, point) == 0:
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

        self.obstructing_walls = np.array([i for i in range(len(self.walls)) if not self.in_convex_hull[i]], dtype=np.int32)


    def plot(self, img_order=None, freq=None, figsize=None, no_axis=False, mic_marker_size=10, **kwargs):
        ''' Plots the room with its walls, microphones, sources and images '''
    
        import matplotlib
        from matplotlib.patches import Circle, Wedge, Polygon
        from matplotlib.collections import PatchCollection
        import matplotlib.pyplot as plt

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

        import matplotlib.pyplot as plt
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

    def first_order_images(self, source_position):

        # projected length onto normal
        ip = np.sum(self.normals * (self.corners - source_position[:, np.newaxis]), axis=0)

        # projected vector from source to wall
        d = ip * self.normals

        # compute images points, positivity is to get only the reflections outside the room
        images = source_position[:, np.newaxis] + 2 * d[:, ip > 0]

        # collect absorption factors of reflecting walls
        damping = (1 - self.absorption[ip > 0])

        # collect the index of the wall corresponding to the new image
        wall_indices = np.arange(len(self.walls))[ip > 0]

        return images, damping, wall_indices


    def image_source_model(self, use_libroom=True):

        self.visibility = []

        for source in self.sources:

            if use_libroom and not libroom_available:
                print("C-extension libroom unavailable. Falling back to pure python")

            # Fall back to pure python if requested or C module unavailable
            if not use_libroom or not libroom_available:
                # Then do it in pure python

                # First, we will generate all the image sources

                # generate first order images
                i, d, w = self.first_order_images(np.array(source.position))
                images = [i]
                damping = [d]
                generators = [-np.ones(i.shape[1])]
                wall_indices = [w]

                # generate all higher order images up to max_order
                o = 1
                while o < self.max_order:
                    # generate all images of images of previous order
                    img = np.zeros((self.dim, 0))
                    dmp = np.array([])
                    gen = np.array([])
                    wal = np.array([])
                    for ind, si, sd in zip(range(images[o-1].shape[1]), images[o - 1].T, damping[o - 1]):
                        i, d, w = self.first_order_images(si)
                        img = np.concatenate((img, i), axis=1)
                        dmp = np.concatenate((dmp, d * sd))
                        gen = np.concatenate((gen, ind*np.ones(i.shape[1])))
                        wal = np.concatenate((wal, w))

                    # sort
                    ordering = np.lexsort(img)
                    img = img[:, ordering]
                    dmp = dmp[ordering]
                    gen = gen[ordering]
                    wal = wal[ordering]

                    if isinstance(self, ShoeBox):
                        '''
                        For shoebox rooms, we can remove duplicate
                        image sources from different wall orderings
                        '''
                        diff = np.diff(img, axis=1)
                        ui = np.ones(img.shape[1], 'bool')
                        ui[1:] = (diff != 0).any(axis=0)

                        # add to array of images
                        images.append(img[:, ui])
                        damping.append(dmp[ui])
                        generators.append(gen[ui])
                        wall_indices.append(wal[ui])

                    else:
                        '''
                        But in general, we have to keep everything
                        '''
                        # add to array of images
                        images.append(img)
                        damping.append(dmp)
                        generators.append(gen)
                        wall_indices.append(wal)

                    # next order
                    o += 1
                    
                o_len = np.array([x.shape[0] for x in generators])
                # correct the pointers for linear structure
                for o in np.arange(2, len(generators)):
                    generators[o] += np.sum(o_len[0:o-1])
                    
                # linearize the arrays
                images_lin = np.concatenate(images, axis=1)
                damping_lin = np.concatenate(damping)
                generators_lin = np.concatenate(generators)
                walls_lin = np.concatenate(wall_indices)
                
                # store the corresponding orders in another array
                ordlist = []
                for o in range(len(generators)):
                    ordlist.append((o+1)*np.ones(o_len[o]))
                orders_lin = np.concatenate(ordlist)

                # add the direct source to the arrays
                source.images = np.concatenate((np.array([source.position]).T, images_lin), axis=1)
                source.damping = np.concatenate(([1], damping_lin))
                source.generators = np.concatenate(([-1], generators_lin+1)).astype(np.int)
                source.walls = np.concatenate(([-1], walls_lin)).astype(np.int)
                source.orders = np.array(np.concatenate(([0], orders_lin)), dtype=np.int)

                # Then we will check the visibilty of the sources
                # visibility is a list with first index for sources, and second for mics
                self.visibility.append([])
                for mic in self.mic_array.R.T:
                    if isinstance(self, ShoeBox):
                        # All sources are visible in shoebox rooms
                        self.visibility[-1].append(np.ones(source.images.shape[1], dtype=bool))
                    else:
                        # In general, we need to check
                        self.visibility[-1].append(
                                self.check_visibility_for_all_images(source, mic, use_libroom=False)
                                )
                self.visibility[-1] = np.array(self.visibility[-1])

                I = np.zeros(self.visibility[-1].shape[1], dtype=bool)
                for mic_vis in self.visibility[-1]:
                    I = np.logical_or(I, mic_vis == 1)

                # Now we can get rid of the superfluous images
                source.images = source.images[:,I]
                source.damping = source.damping[I]
                source.generators = source.generators[I]
                source.walls = source.walls[I]
                source.orders = source.orders[I]

                self.visibility[-1] = self.visibility[-1][:,I]


            else:
                # if libroom is available, use it!

                c_room = self.make_c_room()

                # copy microphone information to struct
                mic = np.asfortranarray(self.mic_array.R, dtype=np.float32)
                c_room.n_microphones = ctypes.c_int(mic.shape[1])
                c_room.microphones = mic.ctypes.data_as(c_float_p)

                src = np.array(source.position, dtype=np.float32)

                if isinstance(self, ShoeBox):

                    # create absorption list in correct order for shoebox algorithm
                    absorption_list_shoebox = np.array([self.absorption_dict[d] for d in self.wall_names], dtype=np.float32)
                    room_dim = np.array(self.shoebox_dim, dtype=np.float32)

                    # Call the dedicated C routine for shoebox room
                    libroom.image_source_shoebox(
                            ctypes.byref(c_room), 
                            src.ctypes.data_as(c_float_p), 
                            room_dim.ctypes.data_as(c_float_p),
                            absorption_list_shoebox.ctypes.data_as(c_float_p),
                            self.max_order
                            )
                else:
                    # Call the general image source generator
                    libroom.image_source_model(ctypes.byref(c_room), src.ctypes.data_as(c_float_p), self.max_order)

                # Recover all the arrays as ndarray from the c struct
                n_sources = c_room.n_sources

                if (n_sources > 0):

                    # numpy wrapper around C arrays
                    images = np.ctypeslib.as_array(c_room.sources, shape=(n_sources, self.dim))
                    orders = np.ctypeslib.as_array(c_room.orders, shape=(n_sources,))
                    gen_walls = np.ctypeslib.as_array(c_room.gen_walls, shape=(n_sources,))
                    attenuations = np.ctypeslib.as_array(c_room.attenuations, shape=(n_sources,))
                    is_visible = np.ctypeslib.as_array(c_room.is_visible, shape=(mic.shape[1], n_sources))

                    # Copy to python managed memory
                    source.images = np.asfortranarray(images.copy().T)
                    source.orders = orders.copy()
                    source.walls = gen_walls.copy()
                    source.damping = attenuations.copy()
                    source.generators = -np.ones(source.walls.shape)

                    self.visibility.append(is_visible.copy())

                    # free the C malloc'ed memory
                    libroom.free_sources(ctypes.byref(c_room))

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
        if len(self.rir) == 0 or recompute_rir:
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

    def print_wall_sequences(self, source):

        visibilityCheck = np.zeros_like(source.images[0])-1
        
        for imageId in range(len(visibilityCheck)-1, -1, -1):
            print("%2d, %d,%.0f,%.0f --- "%(imageId,source.orders[imageId],source.generators[imageId],source.walls[imageId]), end='')
            p = imageId
            while p >= 0:
                if not np.isnan(source.walls[p]):
                    print(int(source.walls[p]), end='')
                p = source.generators[p]
            print()

    def make_c_room(self):
        ''' Wrapper around the C libroom '''

        # exit if libroom is not available
        if not libroom_available:
            return

        # create the ctypes wall array
        c_walls = (CWALL * len(self.walls))()
        c_walls_array = ctypes.cast(c_walls, c_wall_p)
        for cwall, wall in zip(c_walls_array, self.walls):

            c_corners = wall.corners.ctypes.data_as(c_float_p)

            cwall.dim=wall.dim
            cwall.absorption=wall.absorption
            cwall.normal=(ctypes.c_float * 3)(*wall.normal.tolist())
            cwall.n_corners=wall.corners.shape[1]
            cwall.corners=c_corners

            cwall.origin=(ctypes.c_float * 3)(*wall.plane_point.tolist())

            if wall.dim == 3:
                c_corners_2d = wall.corners_2d.ctypes.data_as(c_float_p)

                cwall.basis=(ctypes.c_float * 6)(*wall.plane_basis.flatten('F').tolist())
                cwall.flat_corners=c_corners_2d

        # create the ctypes Room struct
        c_room = CROOM(
                dim = self.dim,
                n_walls = len(self.walls),
                walls = c_walls_array,
                n_obstructing_walls = self.obstructing_walls.shape[0],
                obstructing_walls = self.obstructing_walls.ctypes.data_as(c_int_p),
                )

        return c_room
        
    def check_visibility_for_all_images(self, source, p, use_libroom=True):
        '''
        Checks visibility from a given point for all images of the given source.
        
        This function tests visibility for all images of the source and returns the results
        in an array.
        
        :arg source: (SoundSource) the sound source object (containing all its images)
        :arg p: (np.array size 2 or 3) coordinates of the point where we check visibility
        
        :returns: (int array) list of results of visibility for each image
            -1 : unchecked (only during execution of the function)
            0 (False) : not visible
            1 (True) : visible
        '''
        
        visibilityCheck = np.zeros_like(source.images[0], dtype=np.int32)-1
        
        if self.is_inside(np.array(p)):
            # Only check for points that are in the room!
            if use_libroom and libroom_available:
                # Call the C routine that checks visibility

                # Create the C struct
                c_room = self.make_c_room()

                # copy source information to struct
                c_room.n_sources = ctypes.c_int(source.images.shape[1])
                c_room.sources = source.images.ctypes.data_as(c_float_p)
                c_room.parents = source.generators.ctypes.data_as(c_int_p)
                c_room.gen_walls = source.walls.ctypes.data_as(c_int_p)
                c_room.orders = source.orders.ctypes.data_as(c_int_p)

                # copy microphone information to struct
                mic = np.array(p, dtype=np.float32)
                c_room.n_microphones = ctypes.c_int(1)
                c_room.microphones = mic.ctypes.data_as(c_float_p)

                # add the array for the visibility information
                c_room.is_visible = visibilityCheck.ctypes.data_as(c_int_p)

                # Run the routine here
                libroom.check_visibility_all(ctypes.byref(c_room))

                return visibilityCheck
            else:
                for imageId in range(len(visibilityCheck)-1, -1, -1):
                    visibilityCheck[imageId] = self.is_visible(source, p, imageId)
        else:
            # If point is outside, nothing is visible
            for imageId in range(len(visibilityCheck)-1, -1, -1):
                visibilityCheck[imageId] = False
            
        return visibilityCheck

            
    def is_visible(self, source, p, imageId = 0):
        '''
        Returns true if the given sound source (with image source id) is visible from point p.
        
        :arg source: (SoundSource) the sound source (containing all its images)
        :arg p: (np.array size 2 or 3) coordinates of the point where we check visibility
        :arg imageId: (int) id of the image within the SoundSource object
        
        :return: (bool)
            False (0) : not visible
            True (1) :  visible
        '''

        p = np.array(p)
        imageId = int(imageId)
        
        # Check if there is an obstruction
        if(self.is_obstructed(source, p, imageId)):
            return False
        
        if (source.orders[imageId] > 0):
        
            # Check if the line of sight intersects the generating wall
            genWallId = int(source.walls[imageId])

            # compute the location of the reflection on the wall
            intersection = self.walls[genWallId].intersection(p, np.array(source.images[:, imageId]))[0]

            # the reflection point needs to be visible from the image source that generates the ray
            if intersection is not None:
                    # Check visibility for the parent image by recursion
                    return self.is_visible(source, intersection, source.generators[imageId])
            else:
                return False
        else:
            return True
      
    def is_obstructed(self, source, p, imageId = 0):
        '''
        Checks if there is a wall obstructing the line of sight going from a source to a point.
        
        :arg source: (SoundSource) the sound source (containing all its images)
        :arg p: (np.array size 2 or 3) coordinates of the point where we check obstruction
        :arg imageId: (int) id of the image within the SoundSource object
        
        :returns: (bool)
            False (0) : not obstructed
            True (1) :  obstructed
        '''
        
        imageId = int(imageId)
        if (np.isnan(source.walls[imageId])):
            genWallId = -1
        else:
            genWallId = int(source.walls[imageId])
        
        # Only 'non-convex' walls can be obstructing
        for wallId in self.obstructing_walls:
        
            # The generating wall can't be obstructive
            if(wallId != genWallId):
            
                # Test if the line segment intersects the current wall
                # We ignore intersections at boundaries of the line of sight
                #intersects, borderOfWall, borderOfSegment = self.walls[wallId].intersects(source.images[:, imageId], p)
                intersectionPoint, borderOfSegment, borderOfWall = self.walls[wallId].intersection(source.images[:, imageId], p)

                if (intersectionPoint is not None and not borderOfSegment):
                    
                    # Only images with order > 0 have a generating wall. 
                    # At this point, there is obstruction for source order 0.
                    if (source.orders[imageId] > 0):

                        imageSide = self.walls[genWallId].side(source.images[:, imageId])
                    
                        # Test if the intersection point and the image are at
                        # opposite sides of the generating wall 
                        # We ignore the obstruction if it is inside the
                        # generating wall (it is what happens in a corner)
                        intersectionPointSide = self.walls[genWallId].side(intersectionPoint)
                        if (intersectionPointSide != imageSide and intersectionPointSide != 0):
                            return True
                    else:
                        return True
                
        return False

    def is_inside(self, p, includeBorders = True):
        '''
        Checks if the given point is inside the room.
        
        :arg p: (np.array dim 2 or 3) point to be tested
        :arg includeBorders: (bool) set true if a point on the wall must be considered inside the room
        
        :returns: (bool) True if the given point is inside the room, False otherwise.
        '''
        
        p = np.array(p)
        if (self.dim != p.shape[0]):
            raise ValueError('Dimension of room and p must match.')
        
        # Compute p0, which is a point outside the room at x coordinate xMin-1
        # (where xMin is the minimum x coordinate among the corners of the walls)
        if (self.dim == 2):
            p0 = np.array([np.amin(np.array([wall.corners[0, :] for wall in self.walls]).flatten())-1, p[1]])
        if (self.dim == 3):
            p0 = np.array([np.amin(np.concatenate([wall.corners[0, :] for wall in self.walls]).flatten())-1, p[1], p[2]])
        
        limitCase = False
        count = 0
        for i in range(len(self.walls)):
            intersects, borderOfWall, borderOfSegment = self.walls[i].intersects(p0, p)
            if borderOfSegment:
                limitCase = True
            if intersects:
                count += 1
        if ((not limitCase and count % 2 == 1) or (limitCase and includeBorders)):
            return True
        else:
            return False


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


