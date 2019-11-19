# Utilities for generating random microphone(s) in rooms.
# Copyright (C) 2019  Eric Bezzam
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

import abc
import six
import numpy as np


@six.add_metaclass(abc.ABCMeta)
class Microphone(object):
    """
    Abstract class for sampling a microphone / array in a provided room.
    """
    def __init__(self):
        pass

    @abc.abstractmethod
    def sample(self, room):
        """
        Abstract method to sample a microphone / array within the provided
        room.

        Returns the locations of a microphone / array.

        Parameters
        ----------
        room : Room object
            Room to randomly place a microphone / array inside.
        """
        pass


class OmniMicrophone(Microphone):
    """
    Object to randomly sample an omnidirectional microphone in a provided
    room.

    Parameters
    ----------
    min_dist_wall : float
        Minimum distance from the microphone to each wall.
    min_height : float
        Minimum height of microphone.
    max_height : float
        Maximum height of microphone.
    """
    def __init__(self, min_dist_wall=0.1, min_height=0.4, max_height=1.5):
        Microphone.__init__(self)
        self.min_dist_wall = min_dist_wall
        self.min_height = min_height
        self.max_height = max_height

    def _sample_pos(self, room):
        bbox = room.get_bbox()
        dim = bbox.shape[0]

        count = 0
        while True:
            count += 1

            # sample within bounding box
            sampled_pos = []
            for d in range(dim):
                if d < 2:
                    x = np.random.uniform(
                        bbox[d, 0] + self.min_dist_wall,
                        bbox[d, 1] - self.min_dist_wall
                    )
                else:
                    x = np.random.uniform(
                        max(self.min_height, bbox[d, 0] + self.min_dist_wall),
                        min(self.max_height, bbox[d, 1] - self.min_dist_wall)
                    )
                sampled_pos.append(x)
            sampled_pos = np.array(sampled_pos)

            # check inside room
            if not room.is_inside(sampled_pos):
                continue

            # verify minimum distance to wall
            for k in range(len(room.walls)):
                v_corn = room.walls[k].corners[:, 0] - sampled_pos
                w = room.walls[k].normal
                dist2wall = np.dot(v_corn, w)
                if 0 < dist2wall < self.min_dist_wall:
                    continue
            break

        return sampled_pos

    def sample(self, room):
        return self._sample_pos(room)


class OmniMicrophoneArray(OmniMicrophone):
    """
    Object to randomly sample an array of omnidirectional microphones in a
    provided room.

    Array is assumed to be parallel to the X-Y plane (typically the floor) so
    only 2D coordinates should be provided.

    Parameters
    ----------
    geometry : 2D array
        Array of microphone positions, where each column corresponds to the
        coordinates of a microphone. If not centered, it will be done by
        subtracting the mean.
    min_dist_wall : float
        Minimum distance from the microphone to each wall.
    min_height : float
        Minimum height of microphone.
    max_height : float
        Maximum height of microphone.
    """
    def __init__(self, geometry, min_dist_wall=0.1, min_height=0.4,
                 max_height=1.5):
        assert len(geometry.shape) == 2, 'Only detected one microphone. Use' \
                                         ' `OmniMicrophone` for generating ' \
                                         'single microphone positions.'
        assert geometry.shape[0] == 2, 'Must provide X and Y coordinates ' \
                                       'for microphone positions.'

        # center
        geometry = geometry - geometry.mean(axis=1)[:, np.newaxis]

        # for placing array inside room
        self.max_radius = max(
            np.sqrt(np.diag(np.dot(geometry.T, geometry))))

        # add third dimension
        self.geometry = np.concatenate(
            (geometry, np.zeros((1, geometry.shape[1]))),
            axis=0
        )

        OmniMicrophone.__init__(self,
                                min_dist_wall=min_dist_wall+self.max_radius,
                                min_height=min_height, max_height=max_height)

    def sample(self, room):
        array_center = self._sample_pos(room)
        return self.geometry + array_center[:, np.newaxis]


class DirectionalMicrophone(Microphone):
    def __init__(self):
        raise NotImplementedError