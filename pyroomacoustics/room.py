# @version: 1.0  date: 05/06/2015 by Sidney Barthe
# @author: robin.scheibler@epfl.ch, ivan.dokmanic@epfl.ch, sidney.barthe@epfl.ch
# @copyright: EPFL-IC-LCAV 2015

from __future__ import print_function

import numpy as np
from scipy.spatial import ConvexHull

#import .beamforming as bf
from . import beamforming as bf
from . import geometry as geom
from .soundsource import SoundSource
from .wall import Wall
from .utilities import area, ccw3p
from .parameters import constants, eps


class Room(object):
    """
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
    """

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

        if t0 < (constants.get('frac_delay_length')-1)/float(fs)/2:
            self.t0 = (constants.get('frac_delay_length')-1)/float(fs)/2
        else:
            self.t0 = t0
        
        if (sources is list):
            self.sources = sources
        else:
            self.sources = []

        self.micArray = mics
         
        self.normals = np.array([wall.normal for wall in self.walls]).T
        self.corners = np.array([wall.corners[:, 0] for wall in self.walls]).T
        self.absorption = np.array([wall.absorption for wall in self.walls])

        # Pre-compute RIR if needed
        if (len(self.sources) > 0 and self.micArray is not None):
            self.compute_RIR()
        else:
            self.rir = None
            
        self.dim = walls[0].dim
        self.wallsId = {}
        for i in range(len(walls)):
            if self.walls[i].name is not None:
                self.wallsId[self.walls[i].name] = i

        # check which walls are part of the convex hull
        self.convex_hull()

    @classmethod
    def shoeBox2D(
            cls,
            p1,
            p2,
            absorption=1.,
            fs=8000,
            t0=0.,
            max_order=1,
            sigma2_awgn=None,
            sources=None,
            mics=None):
        """
        Creates a 2D "shoe box" room geometry (rectangle).
        
        :arg p1: (np.array dim 2) coordinates of the lower left corner of the room
        :arg p2: (np.array dim 2) coordinates the upper right corner of the room
        :arg absorption: (float) absorption factor reflection for all walls
        
        :returns: (Room) instance of a 2D shoe box room
        """

        walls = []
        walls.append(Wall(np.array([[p1[0], p2[0]], [p1[1], p1[1]]]), absorption, "south"))
        walls.append(Wall(np.array([[p2[0], p2[0]], [p1[1], p2[1]]]), absorption, "east"))
        walls.append(Wall(np.array([[p2[0], p1[0]], [p2[1], p2[1]]]), absorption, "north"))
        walls.append(Wall(np.array([[p1[0], p1[0]], [p2[1], p1[1]]]), absorption, "west"))

        return cls(walls, fs, t0, max_order, sigma2_awgn, sources, mics)

    @classmethod
    def shoeBox3D(
            cls,
            p1,
            p2,
            absorption=1.,
            fs=8000,
            t0=0.,
            max_order=1,
            sigma2_awgn=None,
            sources=None,
            mics=None):
        """
        Creates a 3D "shoe box" room geometry (rectangular cuboid).
        
        :arg p1: (np.array dim 3) coordinates of the lower left (on floor) corner of the room
        :arg p2: (np.array dim 3) coordinates the upper right (on ceiling) corner of the room
        :arg absorption: (float) absorption factor reflection for all walls
        
        :returns: (Room) instance of a 3D shoe box room
        """

        walls = []
        walls.append(Wall(np.array([[p1[0], p2[0], p2[0], p1[0]], [p1[1], p1[1], p1[1], p1[1]], [p1[2], p1[2], p2[2], p2[2]]]), absorption, "south"))
        walls.append(Wall(np.array([[p2[0], p2[0], p2[0], p2[0]], [p1[1], p2[1], p2[1], p1[1]], [p1[2], p1[2], p2[2], p2[2]]]), absorption, "east"))
        walls.append(Wall(np.array([[p2[0], p1[0], p1[0], p2[0]], [p2[1], p2[1], p2[1], p2[1]], [p1[2], p1[2], p2[2], p2[2]]]), absorption, "north"))
        walls.append(Wall(np.array([[p1[0], p1[0], p1[0], p1[0]], [p2[1], p1[1], p1[1], p2[1]], [p1[2], p1[2], p2[2], p2[2]]]), absorption, "west"))
        walls.append(Wall(np.array([[p2[0], p2[0], p1[0], p1[0]], [p1[1], p2[1], p2[1], p1[1]], [p2[2], p2[2], p2[2], p2[2]]]), absorption, "ceiling"))
        walls.append(Wall(np.array([[p2[0], p1[0], p1[0], p2[0]], [p1[1], p1[1], p2[1], p2[1]], [p1[2], p1[2], p1[2], p1[2]]]), absorption, "floor"))

        return cls(walls, fs, t0, max_order, sigma2_awgn, sources, mics)
        
    @classmethod
    def fromCorners(
            cls,
            corners,
            absorption=1.,
            fs=8000,
            t0=0.,
            max_order=1,
            sigma2_awgn=None,
            sources=None,
            mics=None):
        """
        Creates a 2D room by giving an array of corners.
        
        :arg corners: (np.array dim 2xN, N>2) list of corners, must be antiClockwise oriented
        :arg absorption: (float array or float) list of absorption factor reflection for each wall or single value for all walls
        
        :returns: (Room) instance of a 2D room
        """
        
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
            absorption=1.):
        """
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
        """

        if self.dim != 2:
            raise ValueError('Can only extrude a 2D room.')

        # default orientation vector is pointing up
        if v_vec is None:
            v_vec = np.array([0., 0., 1.])

        # check that the walls are ordered counterclock wise
        # that should be the case if created from fromCorners function
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
                if the room is created with Room.fromCorners")

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
        convex_hull = ConvexHull(X, incremental=True)

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

        self.obstructing_walls = [i for i in range(len(self.walls)) if not self.in_convex_hull[i]]


    def plot(self, img_order=None, freq=None, figsize=None, no_axis=False, mic_marker_size=10, **kwargs):
        """Plots the room with its walls, microphones, sources and images"""
    
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
            if (self.micArray is not None):
                for mic in self.micArray.R.T:
                    ax.scatter(mic[0], mic[1],
                            marker='x', linewidth=0.5, s=mic_marker_size, c='k')

                # draw the beam pattern of the beamformer if requested (and available)
                if freq is not None \
                        and isinstance(self.micArray, bf.Beamformer) \
                        and (self.micArray.weights is not None or self.micArray.filters is not None):

                    freq = np.array(freq)
                    if freq.ndim is 0:
                        freq = np.array([freq])

                    # define a new set of colors for the beam patterns
                    newmap = plt.get_cmap('autumn')
                    desat = 0.7
                    ax.set_color_cycle([newmap(k) for k in desat*np.linspace(0, 1, len(freq))])

                    phis = np.arange(360) * 2 * np.pi / 360.
                    newfreq = np.zeros(freq.shape)
                    H = np.zeros((len(freq), len(phis)), dtype=complex)
                    for i, f in enumerate(freq):
                        newfreq[i], H[i] = self.micArray.response(phis, f)

                    # normalize max amplitude to one
                    H = np.abs(H)**2/np.abs(H).max()**2

                    # a normalization factor according to room size
                    norm = np.linalg.norm((self.corners - self.micArray.center), axis=0).max()

                    # plot all the beam patterns
                    i = 0
                    for f, h in zip(newfreq, H):
                        x = np.cos(phis) * h * norm + self.micArray.center[0, 0]
                        y = np.sin(phis) * h * norm + self.micArray.center[1, 0]
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
            if (self.micArray is not None):
                for mic in self.micArray.R.T:
                    ax.scatter(mic[0], mic[1], mic[2],
                            marker='x', linewidth=0.5, s=mic_marker_size, c='k')


            return fig, ax

    def plotRIR(self, FD=False):

        if self.rir is None:
            self.compute_RIR()

        import matplotlib.pyplot as plt
        import utilities as u

        M = self.micArray.M
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

    def addMicrophoneArray(self, micArray):
        self.micArray = micArray

    def addSource(self, position, signal=None, delay=0):

        if (not self.isInside(np.array(position))):
            raise ValueError('The source must be added inside the room.')

        # generate first order images
        i, d, w = self.firstOrderImages(np.array(position))
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
                i, d, w = self.firstOrderImages(si)
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
        images_lin = np.concatenate((np.array([position]).T, images_lin), axis=1)
        damping_lin = np.concatenate(([1], damping_lin))
        generators_lin = np.concatenate(([-1], generators_lin+1)).astype(np.int)
        walls_lin = np.concatenate(([-1], walls_lin)).astype(np.int)
        orders_lin = np.array(np.concatenate(([0], orders_lin)), dtype=np.int)

        # add a new source to the source list
        self.sources.append(
            SoundSource(
                position,
                images=images_lin,
                damping=damping_lin,
                generators=generators_lin,
                walls=walls_lin,
                orders=orders_lin,
                signal=signal,
                delay=delay))

    def firstOrderImages(self, source_position):

        # projected length onto normal
        ip = np.sum(self.normals * (self.corners - source_position[:, np.newaxis]), axis=0)

        # projected vector from source to wall
        d = ip * self.normals

        # compute images points, positivity is to get only the reflections outside the room
        images = source_position[:, np.newaxis] + 2 * d[:, ip > 0]

        # collect absorption factors of reflecting walls
        damping = self.absorption[ip > 0]

        # collect the index of the wall corresponding to the new image
        wall_indices = np.arange(len(self.walls))[ip > 0]

        return images, damping, wall_indices


    def compute_RIR(self):
        """
        Compute the room impulse response between every source and microphone
        """
        self.rir = []

        for mic in self.micArray.R.T:
            h = []
            for source in self.sources:
                visibility = self.checkVisibilityForAllImages(source, mic)
                h.append(source.getRIR(mic, visibility, self.fs, self.t0))
            self.rir.append(h)

    def simulate(self, recompute_rir=False):
        """Simulates the microphone signal at every microphone in the array"""

        # import convolution routine
        from scipy.signal import fftconvolve

        # Throw an error if we are missing some hardware in the room
        if (len(self.sources) is 0):
            raise ValueError('There are no sound sources in the room.')
        if (self.micArray is None):
            raise ValueError('There is no microphone in the room.')

        # compute RIR if necessary
        if len(self.rir) == 0 or recompute_rir:
            self.compute_RIR()

        # number of mics and sources
        M = self.micArray.M
        S = len(self.sources)

        # compute the maximum signal length
        from itertools import product
        max_len_rir = np.array([len(self.rir[i][j])
                                for i, j in product(range(M), range(S))]).max()
        f = lambda i: len(
            self.sources[i].signal) + np.floor(self.sources[i].delay * self.fs)
        max_sig_len = np.array([f(i) for i in range(S)]).max()
        L = max_len_rir + max_sig_len - 1
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
                d = np.floor(self.sources[s].delay * self.fs)
                h = self.rir[m][s]
                rx[d:d + len(sig) + len(h) - 1] += fftconvolve(h, sig)

            # add white gaussian noise if necessary
            if self.sigma2_awgn is not None:
                rx += np.random.normal(0., np.sqrt(self.sigma2_awgn), rx.shape)

        # record the signals in the microphones
        self.micArray.record(signals, self.fs)


    def dSNR(self, x, source=0):
        """Computes the direct Signal-to-Noise Ratio"""

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


    def getWallFromName(self, name):
        """
        Returns the instance of the wall by giving its name.
        
        :arg name: (string) name of the wall
        
        :returns: (Wall) instance of the wall with this name
        """
        
        if (name in self.wallsId):
            return self.walls[self.wallsId[name]]
        else:
            raise ValueError('The wall '+name+' cannot be found.')

    def printWallSequences(self, source):

        visibilityCheck = np.zeros_like(source.images[0])-1
        
        for imageId in range(len(visibilityCheck)-1, -1, -1):
            print("%2d, %d,%.0f,%.0f --- "%(imageId,source.orders[imageId],source.generators[imageId],source.walls[imageId]), end='')
            p = imageId
            while p >= 0:
                if not np.isnan(source.walls[p]):
                    print(int(source.walls[p]), end='')
                p = source.generators[p]
            print()
        
    def checkVisibilityForAllImages(self, source, p):
        """
        Checks visibility from a given point for all images of the given source.
        
        This function tests visibility for all images of the source and returns the results
        in an array.
        
        :arg source: (SoundSource) the sound source object (containing all its images)
        :arg p: (np.array size 2 or 3) coordinates of the point where we check visibility
        
        :returns: (int array) list of results of visibility for each image
            -1 : unchecked (only during execution of the function)
            0 (False) : not visible
            1 (True) : visible
        """
        
        visibilityCheck = np.zeros_like(source.images[0])-1
        
        if self.isInside(np.array(p)):
            # Only check for points that are in the room!
            for imageId in range(len(visibilityCheck)-1, -1, -1):
                visibilityCheck[imageId] = self.isVisible(source, p, imageId)
        else:
            # If point is outside, nothing is visible
            for imageId in range(len(visibilityCheck)-1, -1, -1):
                visibilityCheck[imageId] = False
            
        return visibilityCheck
            
    def isVisible(self, source, p, imageId = 0):
        """
        Returns true if the given sound source (with image source id) is visible from point p.
        
        :arg source: (SoundSource) the sound source (containing all its images)
        :arg p: (np.array size 2 or 3) coordinates of the point where we check visibility
        :arg imageId: (int) id of the image within the SoundSource object
        
        :return: (bool)
            False (0) : not visible
            True (1) :  visible
        """

        p = np.array(p)
        imageId = int(imageId)
        
        # Check if there is an obstruction
        if(self.isObstructed(source, p, imageId)):
            return False
        
        if (source.orders[imageId] > 0):
        
            # Check if the line of sight intersects the generating wall
            genWallId = int(source.walls[imageId])

            # compute the location of the reflection on the wall
            intersection = self.walls[genWallId].intersection(p, np.array(source.images[:, imageId]))[0]

            # the reflection point needs to be visible from the image source that generates the ray
            if intersection is not None:
                    # Check visibility for the parent image by recursion
                    return self.isVisible(source, intersection, source.generators[imageId])
            else:
                return False
        else:
            return True
      
    def isObstructed(self, source, p, imageId = 0):
        """
        Checks if there is a wall obstructing the line of sight going from a source to a point.
        
        :arg source: (SoundSource) the sound source (containing all its images)
        :arg p: (np.array size 2 or 3) coordinates of the point where we check obstruction
        :arg imageId: (int) id of the image within the SoundSource object
        
        :returns: (bool)
            False (0) : not obstructed
            True (1) :  obstructed
        """
        
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

    def isInside(self, p, includeBorders = True):
        """
        Checks if the given point is inside the room.
        
        :arg p: (np.array dim 2 or 3) point to be tested
        :arg includeBorders: (bool) set true if a point on the wall must be considered inside the room
        
        :returns: (bool) True if the given point is inside the room, False otherwise.
        """
        
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

wall_dict = {'ground':0, 'south':1, 'west':2, 'north':3, 'east':4, 'ceilling':5}

class ShoeBox3D(Room):
    '''
    This class extends room for shoebox room in 3D space.
    '''

    def __init__(self, p1, p2, Fs,
            t0=0.,
            absorption=1.,
            max_order=1,
            sigma2_awgn=None,
            sources=None,
            mics=None):

        self.fs = Fs
        self.t0 = t0

        p1 = np.array(p1, dtype='float64')
        p2 = np.array(p2, dtype='float64')

        if p1.shape[0] != 3 or p2.shape[0] != 3:
            raise NameError('Defining points must have 3 elements each.')
        if p1.ndim != 1 or p2.ndim != 1:
            raise NameError('Defining points must be 1 dimensional.')

        # We order the faces as ground first and ceiling last
        # walls order: [Ground, South, West, North, East, Ceilling]
        # where South: Wall alligned with x axis with least y coordinate
        #       West : Wall alligned with y axis with least x coordinate
        #       North: Wall alligned with x axis with largest y coordinate
        #       East : Wall alligned with y axis with largest x coordinate
        self.dim = 3.
        self.corners = np.array([p1, p1, p1, p2, p2, p2]).T
        self.normals = np.array([[ 0.,  0., -1.,  0., 1., 0.],
                                 [ 0., -1.,  0.,  1., 0., 0.],
                                 [-1.,  0.,  0.,  0., 0., 1.]])

        # Array of walls. This is a hack.
        # For 2D rooms, every wall is a 2D vector.
        # To generalize to 3D room this will need to be changed
        # to a list of Wall objects. Wall objects are 2D or 3D polygons.
        # For now, we just need the array to have self.walls.shape[1]
        # to be defined because it is used in firstOrderImages function
        self.walls = np.zeros(self.normals.shape)

        # list of attenuation factors for the wall reflections
        if isinstance(absorption, dict):
            self.absorption = np.zeros(self.normals.shape[1])
            for key,val in absorption.iteritems():
                try:
                    self.absorption[wall_dict[key]] = val
                except KeyError:
                    print('Warning: non-existent wall name. Ignoring.')
        else:
            absorption = np.array(absorption, dtype='float64')
            if (absorption.ndim == 0):
                self.absorption = absorption * np.ones(self.corners.shape[1])
            elif (absorption.ndim > 1 or self.corners.shape[1] != absorption.shape[0]):
                raise NameError('Absorption and corner must be the same size')
            else:
                self.absorption = absorption

        # a list of sources
        if (sources is None):
            self.sources = []
        elif (sources is list):
            self.sources = sources
        else:
            raise NameError('Room needs a source or list of sources.')

        # a microphone array
        if (mics is not None):
            self.micArray = None
        else:
            self.micArray = mics

        # a maximum orders for image source computation
        self.max_order = max_order

        # pre-compute RIR if needed
        if (len(self.sources) > 0 and self.micArray is not None):
            self.compute_RIR()
        else:
            self.rir = []

        # ambiant additive white gaussian noise level
        self.sigma2_awgn = sigma2_awgn


