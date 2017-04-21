# @version: 1.0  date: 05/06/2015 by Sidney Barthe
# @author: robin.scheibler@epfl.ch, ivan.dokmanic@epfl.ch, sidney.barthe@epfl.ch
# @copyright: EPFL-IC-LCAV 2015
from __future__ import division, print_function

import numpy as np

from .parameters import constants


class SoundSource(object):
    '''
    A class to represent sound sources.
    
    This object represents a sound source in a room by a list containing the original source position
    as well as all the image sources, up to some maximum order.

    It also keeps track of the sequence of generated images and the index of the walls (in the original room)
    that generated the reflection.
    '''

    def __init__(
            self,
            position,
            images=None, #source position
            damping=None,
            generators=None, #parent source
            walls=None, #generating wall
            orders=None,
            signal=None,
            delay=0):

        self.position = np.array(position)
        self.dim = self.position.shape[0]

        if (images is None):
            # set to empty list if nothing provided
            self.images = np.asfortranarray(np.array([position], dtype=np.float32).T)
            self.damping = np.array([1.])
            self.generators = np.array([-1], dtype=np.int32)
            self.walls = np.array([-1], dtype=np.int32)
            self.orders = np.array([0], dtype=np.int32)

        else:
            # we need to have damping factors for every image
            if (damping is None):
                # set to one if not set
                damping = np.ones(images.shape[1])

            if images.shape[1] != damping.shape[0]:
                raise NameError('Images and damping must have same shape')

            if generators is not None and generators.shape[0] != images.shape[1]:
                raise NameError('Images and generators must have same shape')

            if walls is not None and walls.shape[0] != images.shape[1]:
                raise NameError('Images and walls must have same shape')

            if orders is not None and orders.shape[0] != images.shape[1]:
                raise NameError('Images and orders must have same shape')


            self.images = np.array(images, order='F', dtype=np.float32)
            self.damping = damping
            self.walls = np.array(walls, dtype=np.int32)
            self.generators = np.array(generators, dtype=np.int32)
            self.orders = np.array(orders, dtype=np.int32)

        # store the natural ordering for the images
        self.I = np.arange(self.images.shape[1])

        # the natural ordering is per generation
        self.ordering = 'order'

        # The sound signal of the source
        self.signal = signal
        self.delay = delay
        self.max_order = np.max(self.orders)


    def add_signal(signal):

        self.signal = signal

    def distance(self, ref_point):

        return np.sqrt(np.sum((self.images - ref_point[:, np.newaxis])**2, axis=0))

    def set_ordering(self, ordering, ref_point=None):
        '''
        Set the order in which we retrieve images sources.
        Can be: 'nearest', 'strongest', 'order'
        Optional argument: ref_point
        '''

        self.ordering = ordering

        if ref_point is not None and ref_point.ndim > 1:
            ref_point = ref_point[:, 0]

        if ordering == 'nearest':

            if ref_point is None:
                raise NameError('For nearest ordering, a reference point is needed.')

            self.I = self.distance(ref_point).argsort()

        elif ordering == 'strongest':

            if ref_point is None:
                raise NameError('For strongest ordering, a reference point is needed.')

            strength = self.damping / (4*np.pi*self.distance(ref_point))
            self.I = strength.argsort()

        elif ordering == 'order':

            self.ordering = 'order'

        else:
            raise NameError('Ordering can be nearest, strongest, order.')


    def __getitem__(self, index):
        '''Overload the bracket operator to access a subset image sources'''

        if isinstance(index, slice) or isinstance(index, int):
            if self.ordering == 'order':
                p_orders = np.arange(0, self.max_order+1)[index]
                # we use the any operator and broadcasting to get match on
                # all image source of order contained in p_orders
                I = np.any(self.orders[:, np.newaxis] == p_orders[np.newaxis, :], axis=1)
                s = SoundSource(
                    self.position,
                    images=self.images[:, I],
                    damping=self.damping[I],
                    orders=self.orders[I],
                    signal=self.signal,
                    delay=self.delay,
                    generators=self.generators[I],
                    walls=self.walls[I])
            else:
                s = SoundSource(
                    self.position,
                    images=self.images[:, self.I[index]],
                    damping=self.damping[self.I[index]],
                    orders=self.orders[self.I[index]],
                    signal=self.signal,
                    delay=self.delay,
                    generators=self.generators[self.I[index]],
                    walls=self.walls[self.I[index]])
        else:
            s = SoundSource(
                self.position,
                images=self.images[:, index],
                damping=self.damping[index],
                orders=self.orders[index],
                signal=self.signal,
                delay=self.delay,
                generators=self.generators[index],
                walls=self.walls[index])

        return s


    def get_images(self, max_order=None, max_distance=None, n_nearest=None, ref_point=None):
        '''
        Keep this for compatibility
        Now replaced by the bracket operator and the setOrdering function.
        '''

        # TO DO: Add also n_strongest

        # TO DO: Make some of these thing exclusive (e.g. can't have n_nearest
        # AND n_strongest (although could have max_order AND n_nearest)

        # TO DO: Make this more efficient if bottleneck (unlikely)

        if (max_order is None):
            max_order = np.max(self.orders)

        # stack source and all images
        I_ord = (self.orders <= max_order)
        img = self.images[:, I_ord]

        if (n_nearest is not None):
            dist = np.sum((img - ref_point)**2, axis=0)
            I_near = dist.argsort()[0:n_nearest]
            img = img[:, I_near]

        return img


    def get_damping(self, max_order=None):
        if (max_order is None):
            max_order = len(np.max(self.orders))

        return self.damping[self.orders <= max_order]


    def get_rir(self, mic, visibility, Fs, t0=0., t_max=None):
        '''
        Compute the room impulse response between the source
        and the microphone whose position is given as an
        argument.
        '''

        # fractional delay length
        fdl = constants.get('frac_delay_length')
        fdl2 = (fdl-1) // 2

        # compute the distance
        dist = self.distance(mic)
        time = dist / constants.get('c') + t0
        alpha = self.damping / (4.*np.pi*dist)

        # the number of samples needed
        if t_max is None:
            # we give a little bit of time to the sinc to decay anyway
            N = np.ceil((1.05*time.max() - t0) * Fs)
        else:
            N = np.ceil((t_max - t0) * Fs)

        N += fdl

        t = np.arange(N) / float(Fs)
        ir = np.zeros(t.shape)

        # from utilities import lowPassDirac
        from .utilities import fractional_delay
        #return u.lowPassDirac(time[:, np.newaxis], alpha[:, np.newaxis], Fs, N).sum(axis=0)

        for i in range(time.shape[0]):
            if visibility[i] == 1:
                time_ip = int(np.round(Fs * time[i]))
                time_fp = (Fs * time[i]) - time_ip
                ir[time_ip-fdl2:time_ip+fdl2+1] += alpha[i]*fractional_delay(time_fp)

        return ir

    def wall_sequence(self,i):
        '''
        Print the wall sequence for the image source indexed by i
        '''
        p = self.generators[i]
        if np.isnan(p):
            return []
        w = [self.walls[i]]
        while not np.isnan(p):
            w.append(self.walls[p])
            p = self.generators[p]
        return w


def build_rir_matrix(mics, sources, Lg, Fs, epsilon=5e-3, unit_damping=False):
    '''
    A function to build the channel matrix for many sources and microphones

    mics is a dim-by-M ndarray where each column is the position of a microphone
    sources is a list of SoundSource objects
    Lg is the length of the beamforming filters
    Fs is the sampling frequency
    epsilon determines how long the sinc is let to decay. Defaults to epsilon=5e-3
    unit_damping determines if the wall damping parameters are used or not. Default to false.

    returns the RIR matrix H =

    ::

         --------------------
         | H_{11} H_{12} ...
         | ...
         |
         --------------------

    where H_{ij} is channel matrix between microphone i and source j.
    H is of type (M*Lg)x((Lg+Lh-1)*S) where Lh is the channel length (determined by epsilon),
    and M, S are the number of microphones, sources, respectively.
    '''

    from .beamforming import distance
    from .utilities import low_pass_dirac, convmtx

    # set the boundaries of RIR filter for given epsilon
    d_min = np.inf
    d_max = 0.
    dmp_max = 0.
    for s in range(len(sources)):
        dist_mat = distance(mics, sources[s].images)
        if unit_damping is True:
            dmp_max = np.maximum((1. / (4*np.pi*dist_mat)).max(), dmp_max)
        else:
            dmp_max = np.maximum((sources[s].damping[np.newaxis, :] / (4*np.pi*dist_mat)).max(), dmp_max)
        d_min = np.minimum(dist_mat.min(), d_min)
        d_max = np.maximum(dist_mat.max(), d_max)

    t_max = d_max / constants.get('c')
    t_min = d_min / constants.get('c')
        
    offset = dmp_max / (np.pi*Fs*epsilon)

    # RIR length
    Lh = int((t_max - t_min + 2*offset)*float(Fs))

    # build the channel matrix
    L = Lg + Lh - 1
    H = np.zeros((Lg*mics.shape[1], len(sources)*L))

    for s in range(len(sources)):
        for r in np.arange(mics.shape[1]):

            dist = sources[s].distance(mics[:,r])
            time = dist / constants.get('c') - t_min + offset
            if unit_damping == True:
                dmp = 1. / (4*np.pi*dist)
            else:
                dmp = sources[s].damping / (4*np.pi*dist)

            h = low_pass_dirac(time[:, np.newaxis], dmp[:, np.newaxis], Fs, Lh).sum(axis=0)
            H[r*Lg:(r+1)*Lg, s*L:(s+1)*L] = convmtx(h, Lg).T

    return H
