
import numpy as np

import beamforming as bf
from SoundSource import SoundSource

import constants

'''
Room
A room geometry is defined by all the sources and all their images
'''

class Room(object):

    def __init__(
            self,
            corners,
            Fs,
            t0=0.,
            absorption=1.,
            max_order=1,
            sigma2_awgn=None,
            sources=None,
            mics=None):

        # make sure we have an ndarray of the right size
        corners = np.array(corners)
        if (np.rank(corners) != 2):
            raise NameError('Room corners is a 2D array.')

        # make sure the corners are anti-clockwise
        if (self.area(corners) <= 0):
            raise NameError('Room corners must be anti-clockwise')

        self.corners = corners
        self.dim = corners.shape[0]

        # sampling frequency and time offset
        self.Fs = Fs
        self.t0 = t0

        # circular wall vectors (counter clockwise)
        self.walls = self.corners - \
            self.corners[:, xrange(-1, corners.shape[1] - 1)]

        # compute normals (outward pointing)
        self.normals = self.walls[[1, 0], :]/np.linalg.norm(self.walls, axis=0)[np.newaxis,:]
        self.normals[1, :] *= -1;

        # list of attenuation factors for the wall reflections
        absorption = np.array(absorption, dtype='float64')
        if (np.rank(absorption) == 0):
            self.absorption = absorption * np.ones(self.corners.shape[1])
        elif (np.rank(absorption) > 1 or self.corners.shape[1] != len(absorption)):
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


    def plot(self, img_order=None, freq=None, figsize=None, no_axis=False, mic_marker_size=10, **kwargs):

        import matplotlib
        from matplotlib.patches import Circle, Wedge, Polygon
        from matplotlib.collections import PatchCollection
        import matplotlib.pyplot as plt

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
        #if no_axis is True:
            #r = Rectangle((xlim[0],ylim[0]), xlim[1]-xlim[0], ylim[1]-ylim[0])
            #polygons.append(r)
        p = PatchCollection(polygons, cmap=matplotlib.cm.jet, 
                facecolor=np.array([1,1,1]), edgecolor=np.array([0,0,0]))
        ax.add_collection(p)

        # draw the microphones
        if (self.micArray is not None):
            for mic in self.micArray.R.T:
                ax.scatter(mic[0], mic[1], 
                        marker='x', linewidth=0.5, s=mic_marker_size, c='k')

            # draw the beam pattern of the beamformer if requested (and
            # available)
            if freq is not None \
                    and isinstance(self.micArray, bf.Beamformer) \
                    and (self.micArray.weights is not None 
                            or self.micArray.filters is not None):

                freq = np.array(freq)
                if np.rank(freq) is 0:
                    freq = np.array([freq])

                # define a new set of colors for the beam patterns
                newmap = plt.get_cmap('autumn')
                desat = 0.7
                ax.set_color_cycle([newmap(k) for k in desat*np.linspace(0,1,len(freq))])

                
                phis = np.arange(360) * 2 * np.pi / 360.
                newfreq = np.zeros(freq.shape)
                H = np.zeros((len(freq), len(phis)), dtype=complex)
                for i,f in enumerate(freq):
                    newfreq[i], H[i] = self.micArray.response(phis, f)

                # normalize max amplitude to one
                H = np.abs(H)**2/np.abs(H).max()**2

                # a normalization factor according to room size
                norm = np.linalg.norm(
                        (self.corners - self.micArray.center),
                        axis=0).max()
                    
                # plot all the beam patterns
                i = 0
                for f,h in zip(newfreq, H):
                    x = np.cos(phis) * h * norm + self.micArray.center[0, 0]
                    y = np.sin(phis) * h * norm + self.micArray.center[1, 0]
                    l = ax.plot(x, y, '-', linewidth=0.5)
                    #lbl = '%.2f' % f
                    #i0 = i*360/len(freq)
                    #ax.text(x[i0], y[i0], lbl, color=plt.getp(l[0], 'color'))
                    #i += 1

                #ax.legend(freq)

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
                marker=markers[
                    i %
                    len(markers)],
                edgecolor=cmap(1.))
            #ax.text(source.position[0]+0.1, source.position[1]+0.1, str(i))

            # draw images
            if (img_order is None):
                img_order = self.max_order
            for o in xrange(img_order):
                # map the damping to a log scale (mapping 1 to 1)
                val = (np.log2(source.damping[o]) + 10.) / 10.
                # plot the images
                ax.scatter(source.images[o][0, :], source.images[o][1,:], \
                           c=cmap(val), s=20,
                           marker=markers[i % len(markers)], edgecolor=cmap(val))

        # keep axis equal, or the symmetry is lost
        #ax.axis('equal')

        return fig, ax

    def plotRIR(self, FD=False):

        if self.rir == None:
            self.compute_RIR()

        import matplotlib.pyplot as plt
        import utilities as u

        M = self.micArray.M
        S = len(self.sources)
        for r in xrange(M):
            for s in xrange(S):
                h = self.rir[r][s]
                plt.subplot(M, S, r*S + s + 1)
                if not FD:
                    plt.plot(np.arange(len(h)) / float(self.Fs), h)
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

        # generate first order images
        i, d = self.firstOrderImages(np.array(position))
        images = [i]
        damping = [d]

        # generate all higher order images up to max_order
        o = 1
        while o < self.max_order:
            # generate all images of images of previous order
            img = np.zeros((self.dim, 0))
            dmp = np.array([])
            for si, sd in zip(images[o - 1].T, damping[o - 1]):
                i, d = self.firstOrderImages(si)
                img = np.concatenate((img, i), axis=1)
                dmp = np.concatenate((dmp, d * sd))

            # remove duplicates
            ordering = np.lexsort(img)
            img = img[:, ordering]
            dmp = dmp[ordering]
            diff = np.diff(img, axis=1)
            ui = np.ones(img.shape[1], 'bool')
            ui[1:] = (diff != 0).any(axis=0)

            # add to array of images
            images.append(img[:, ui])
            damping.append(dmp[ui])

            # next order
            o += 1

        # add a new source to the source list
        self.sources.append(
            SoundSource(
                position,
                images=images,
                damping=damping,
                signal=signal,
                delay=delay))

    def firstOrderImages(self, source_position):

        # projected length onto normal
        ip = np.sum(
            self.normals * (self.corners - source_position[:, np.newaxis]), axis=0)

        # projected vector from source to wall
        d = ip * self.normals

        # compute images points, positivity is to get only the reflections
        # outside the room
        images = source_position[:, np.newaxis] + 2 * d[:, ip > 0]

        # collect absorption factors of reflecting walls
        damping = self.absorption[ip > 0]

        return images, damping


    def compute_RIR(self, c=constants.c, window=False):
        '''
        Compute the room impulse response between every source and microphone
        '''
        self.rir = []

        for mic in self.micArray.R.T:

            h = []

            for source in self.sources:

                # stack source and all images
                img = source.getImages(self.max_order)
                dmp = source.getDamping(self.max_order)

                # compute the distance
                dist = np.sqrt(np.sum((img - mic[:, np.newaxis]) ** 2, axis=0))
                time = dist / c + self.t0
                alpha = dmp/(4.*np.pi*dist)

                # the number of samples needed
                N = np.ceil((time.max() + self.t0) * self.Fs)

                t = np.arange(N)/float(self.Fs)
                ir = np.zeros(t.shape)

                #from utilities import lowPassDirac
                import utilities as u

                ir = u.lowPassDirac(time[:,np.newaxis], alpha[:,np.newaxis], self.Fs, N).sum(axis=0)

                #for ti, ai in zip(time, alpha):
                    #ir += u.lowPassDirac(ti, ai, self.Fs, N)

                h.append(ir)

            self.rir.append(h)


    def simulate(self, recompute_rir=False):
        '''
        Simulate the microphone signal at every microphone in the array
        '''

        # import convolution routine
        from scipy.signal import fftconvolve

        # Throw an error if we are missing some hardware in the room
        if (len(self.sources) is 0):
            raise NameError('There are no sound sources in the room.')
        if (self.micArray is None):
            raise NameError('There is no microphone in the room.')

        # compute RIR if necessary
        if len(self.rir) == 0 or recompute_rir:
            self.compute_RIR()

        # number of mics and sources
        M = self.micArray.M
        S = len(self.sources)

        # compute the maximum signal length
        from itertools import product
        max_len_rir = np.array([len(self.rir[i][j])
                                for i, j in product(xrange(M), xrange(S))]).max()
        f = lambda i: len(
            self.sources[i].signal) + np.floor(self.sources[i].delay * self.Fs)
        max_sig_len = np.array([f(i) for i in xrange(S)]).max()
        L = max_len_rir + max_sig_len - 1
        if L%2 == 1: L += 1

        # the array that will receive all the signals
        self.micArray.signals = np.zeros((M, L))

        # compute the signal at every microphone in the array
        for m in np.arange(M):
            rx = self.micArray.signals[m]
            for s in np.arange(S):
                sig = self.sources[s].signal
                if sig is None:
                    continue
                d = np.floor(self.sources[s].delay * self.Fs)
                h = self.rir[m][s]
                rx[d:d + len(sig) + len(h) - 1] += fftconvolve(h, sig)

            # add white gaussian noise if necessary
            if self.sigma2_awgn is not None:
                rx += np.random.normal(0., np.sqrt(self.sigma2_awgn), rx.shape)


    def dSNR(self, x, source=0):
        ''' direct Signal-to-Noise Ratio'''

        if source >= len(self.sources):
            raise NameError('No such source')

        if self.sources[source].signal is None:
            raise NameError('No signal defined for source ' + str(source))

        if self.sigma2_awgn is None:
            return float('inf')

        x = np.array(x)

        sigma2_s = np.mean(self.sources[0].signal**2)

        d2 = np.sum((x - self.sources[source].position)**2)

        return sigma2_s/self.sigma2_awgn/(16*np.pi**2*d2)


    @classmethod
    def shoeBox2D(cls, p1, p2, Fs, **kwargs):
        '''
        Create a new Shoe Box room geometry.
        Arguments:
        p1: the lower left corner of the room
        p2: the upper right corner of the room
        max_order: the maximum order of image sources desired.
        '''

        # compute room characteristics
        corners = np.array(
            [[p1[0], p2[0], p2[0], p1[0]], [p1[1], p1[1], p2[1], p2[1]]])

        return Room(corners, Fs, **kwargs)

    @classmethod
    def area(cls, corners):
        '''
        Compute the area of a 2D room represented by its corners
        '''
        x = corners[0, :] - corners[0, xrange(-1, corners.shape[1]-1)]
        y = corners[1, :] + corners[1, xrange(-1, corners.shape[1]-1)]
        return -0.5 * (x * y).sum()

    @classmethod
    def isAntiClockwise(cls, corners):
        '''
        Return true if the corners of the room are arranged anti-clockwise
        '''
        return (cls.area(corners) > 0)

    @classmethod
    def ccw3p(cls, p):
        '''
        Argument: p, a (3,2)-ndarray whose rows are the vertices of a 2D triangle
        Returns
        1: if triangle vertices are counter-clockwise
        -1: if triangle vertices are clock-wise
        0: if vertices are colinear

        Ref: https://en.wikipedia.org/wiki/Curve_orientation
        '''
        if (p.shape != (2, 3)):
            raise NameError(
                'Room.ccw3p is for three 2D points, input is 3x2 ndarray')
        D = (p[0, 1] - p[0, 0]) * (p[1, 2] - p[1, 0]) - \
            (p[0, 2] - p[0, 0]) * (p[1, 1] - p[1, 0])

        if (np.abs(D) < constants.eps):
            return 0
        elif (D > 0):
            return 1
        else:
            return -1
