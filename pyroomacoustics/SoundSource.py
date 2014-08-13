
import numpy as np

'''
A class to represent sound sources
'''


class SoundSource(object):

    def __init__(
            self,
            position,
            images=None,
            damping=None,
            signal=None,
            delay=0):

        self.position = np.array(position)

        if (images is None):
            # set to empty list if nothing provided
            self.images = []
            self.damping = []

        else:
            # save list if provided
            self.images = images

            # we need to have damping factors for every image
            if (damping is None):
                # set to one if not set
                self.damping = []
                for o in images:
                    self.damping.append(np.ones(o.shape))
            else:
                # check damping is the same size as images
                if (len(damping) != len(images)):
                    raise NameError('Images and damping must have same shape')
                for i in range(len(damping)):
                    if (damping[i].shape[0] != images[i].shape[1]):
                        raise NameError(
                            'Images and damping must have same shape')

                # copy over if correct
                self.damping = damping

        # The sound signal of the source
        self.signal = signal
        self.delay = delay

    def addSignal(signal):

        self.signal = signal

    def getImages(self, max_order=None, max_distance=None, n_nearest=None, ref_point=None):

        # TO DO: Add also n_strongest

        # TO DO: Make some of these thing exclusive (e.g. can't have n_nearest
        # AND n_strongest (although could have max_order AND n_nearest)

        # TO DO: Make this more efficient if bottleneck (unlikely)

        if (max_order is None):
            max_order = len(self.images)

        # stack source and all images
        img = np.array([self.position]).T
        for o in xrange(max_order):
            img = np.concatenate((img, self.images[o]), axis=1)

        if (n_nearest is not None):
            dist = np.sum((img - ref_point)**2, axis=0)
            i_nearest = dist.argsort()[0:n_nearest]
            img = img[:,i_nearest]

        return img

    def getDamping(self, max_order=None):
        if (max_order is None):
            max_order = len(images)

        # stack source and all images
        dmp = np.array([1.])
        for o in xrange(max_order):
            dmp = np.concatenate((dmp, self.damping[o]))

        return dmp
