# Utility functions generating a dataset of room impulse responses.
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

import numpy as np
import datetime
import os
import uuid
import json

import pyroomacoustics as pra
from pyroomacoustics.random.distribution import DiscreteDistribution, \
    MultiUniformDistribution, UniformDistribution
from pyroomacoustics.doa.utils import spher2cart
from pyroomacoustics.room import ShoeBox

"""
Utility functions for creating a dataset of room impulse responses.

See `examples/generate_room_dataset.py` for an example of creating a dataset 
of room impulse responses and applying them.
"""


class RoomSimulationDistributions(object):
    """
    Largely based off of distributions presented in Section 2.1 of
    `this paper <https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46107.pdf>`_.
    """

    snr = DiscreteDistribution(
        values=[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
        prob=[6, 10, 14.5, 16.5, 16.5, 14.5, 10, 6, 2.5, 1, 0.5]
    )
    room_dimensions = MultiUniformDistribution(
        # width, length, height
        ranges=[[3, 10], [3, 8], [2.5, 6]]
    )
    target_orientation = MultiUniformDistribution(
        # azimuth, elevation
        ranges=[[-180, 180], [45, 135]]
    )
    target_mic_dist = DiscreteDistribution(
        values=[1, 2, 3, 4, 5, 6, 7],
        prob=[15, 22, 29, 21, 8, 3, 0.5]
    )
    noise_mic_dist = DiscreteDistribution(
        values=[1, 2, 3, 4, 5, 6, 7],
        prob=[15, 22, 29, 21, 8, 3, 0.5]
    )
    n_noise = DiscreteDistribution(
        values=[0, 1, 2, 3],
        prob=[1, 1, 1, 1]
    )
    noise_orientation = MultiUniformDistribution(
        # azimuth, elevation
        ranges=[[-180, 180], [-30, 180]]
    )
    rt60 = DiscreteDistribution(
        # reverberation time
        values=[0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        prob=[4, 6, 7.5, 10, 12.5, 16.5, 16.5, 12.5, 7.5, 4]
    )
    mic_height = UniformDistribution(vals_range=(1., 1.5))


class ShoeBoxRoomGenerator(object):
    """

    `ShoeBox` room generator.

    Default distributions correspond to parameters from
    `this paper <https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46107.pdf>`_.

    Parameters
    -----------
    room_dim_distrib : `MultiUniformDistribution` or `MultiDiscreteDistribution`
        Multi-dimensional distribution for (width, length, height) of ShoeBox
        room.
    target_orientation_distrib : `MultiUniformDistribution` or `MultiDiscreteDistribution`
        Multi-dimensional distribution for (azimuth, elevation) angles of
        target source.
    target_mic_dist_distrib : `UniformDistribution` or `DiscreteDistribution`
        Single dimensional distribution for distance between target and
        microphone(s).
    n_noise_distrib : `DiscreteDistribution`
        Discrete distribution for number of noise sources.
    noise_orientation_distrib : `MultiUniformDistribution` or `MultiDiscreteDistribution`
        Multi-dimensional distribution for (azimuth, elevation) angles of
        noise source.
    noise_mic_dist_distrib : `UniformDistribution` or `DiscreteDistribution`
        Single dimensional distribution for distance between noise and
        microphone(s).
    rt60_distrib : `UniformDistribution` or `DiscreteDistribution`
        Single dimensional distribution for average reverberation time of room.
    mic_height_distrib : `UniformDistribution` or `DiscreteDistribution`
        Single dimensional distribution for height of microphone(s).
    mic_min_dist_wall : float
        Minimum distance between microphone(s) and each wall. Default is 0.1
        meters.
    source_min_dist_wall : float
        Minimum distance between microphone(s) and each source. Default is 0.5
        meters.
    sample_rate : int
        Sample rate in Hz. Default is 16000 Hz.
    ism_order : int
        Image Source Method order for RIR generation. Default is 17.
    air_absorption: bool, optional
        If set to True, absorption of sound energy by the air will be
        simulated.
    ray_tracing: bool, optional
        If set to True, the ray tracing simulator will be used along with
        image source model.
    timeout : int
        Number of times to try generating a room with the desired configuration
        until giving up. Default is 1000.
    """

    def __init__(self,
                 room_dim_distrib=None,
                 target_orientation_distrib=None,
                 target_mic_dist_distrib=None,
                 noise_mic_dist_distrib=None,
                 noise_orientation_distrib=None,
                 n_noise_distrib=None,
                 rt60_distrib=None,
                 mic_height_distrib=None,
                 mic_min_dist_wall=0.1,
                 source_min_dist_wall=0.5,
                 source_min_dist_mic=0.1,
                 sample_rate=1source_min_dist_mic6000,
                 ism_order=17,
                 air_absorption=True,
                 ray_tracing=True,
                 timeout=1000,
                 ):

        self.room_dim_distrib = room_dim_distrib \
            if room_dim_distrib is not None \
            else RoomSimulationDistributions.room_dimensions
        self.target_orientation_distrib = target_orientation_distrib \
            if target_orientation_distrib is not None \
            else RoomSimulationDistributions.target_orientation
        self.target_mic_dist_distrib = target_mic_dist_distrib \
            if target_mic_dist_distrib is not None \
            else RoomSimulationDistributions.target_mic_dist
        self.noise_mic_dist_distrib = noise_mic_dist_distrib \
            if noise_mic_dist_distrib is not None \
            else RoomSimulationDistributions.noise_mic_dist
        self.noise_orientation_distrib = noise_orientation_distrib \
            if noise_orientation_distrib is not None \
            else RoomSimulationDistributions.noise_orientation
        self.n_noise_distrib = n_noise_distrib \
            if n_noise_distrib is not None \
            else RoomSimulationDistributions.n_noise
        self.rt60_distrib = rt60_distrib if rt60_distrib is not None \
            else RoomSimulationDistributions.rt60
        self.mic_height_distrib = mic_height_distrib \
            if mic_height_distrib is not None \
            else RoomSimulationDistributions.mic_height

        self.mic_min_dist_wall = mic_min_dist_wall
        self.source_min_dist_wall = source_min_dist_wall
        self.source_min_dist_mic = source_min_dist_mic
        self.sample_rate = sample_rate
        self.ism_order = ism_order
        self.air_absorption = air_absorption
        self.ray_tracing = ray_tracing
        self.timeout = timeout

    def create_dataset(self, n_rooms, output_dir=None):
        """
        Create a dataset of room impulse responses with the following
        structure::

            <output_dir>/
                room_metadata.json
                data/
                    room_<uuid>.npz
                    room_<uuid>.npz
                    ...

        Parameters
        ----------
        n_rooms : int
            Number of rooms to generate.
        output_dir : str, optional
            Path to place create dataset. If not provided, the dataset will
            be created in the same directory as where the script is called
            and with a timestamp as part of the directory.

        """

        if output_dir is None:
            ts = datetime.datetime.now()
            output_dir = 'pra_room_dataset_{}'.format(
                ts.strftime('%Y-%m-%d-%Hh%Mm%Ss'))

        # create output directory
        os.mkdir(output_dir)
        print('Created output directory : {}'.format(output_dir))
        data_dir = os.path.join(output_dir, 'data')
        os.mkdir(data_dir)

        # sample rooms
        room_metadata = dict()
        for _ in range(n_rooms):

            count = 0
            while count < self.timeout:

                try:

                    # sample room
                    room_dim = self.room_dim_distrib.sample()

                    # sample absorption factor / material
                    rt60 = self.rt60_distrib.sample()
                    energy_absorp = rt60_to_absorption_eyring(room_dim, rt60)
                    materials = pra.Material.make_freq_flat(
                        absorption=energy_absorp
                    )

                    # mic location
                    mic_height = self.mic_height_distrib.sample()
                    assert mic_height > self.mic_min_dist_wall
                    assert mic_height < room_dim[2] - self.mic_min_dist_wall
                    mic_loc = sample_mic_location(room_dim,
                                                  self.mic_min_dist_wall)
                    mic_loc.append(mic_height)

                    # create Room object
                    R = np.array(mic_loc, ndmin=2).T
                    room = ShoeBox(p=room_dim,
                                   fs=self.sample_rate,
                                   materials=materials,
                                   max_order=self.ism_order,
                                   mics=pra.MicrophoneArray(R,
                                                            self.sample_rate),
                                   air_absorption=self.air_absorption,
                                   ray_tracing=self.ray_tracing,
                                   )

                    # sample target location
                    target_orientation = \
                        self.target_orientation_distrib.sample()
                    target_dist = self.target_mic_dist_distrib.sample()
                    if target_dist < self.source_min_dist_mic:
                        count += 1
                        continue
                    target_loc = mic_loc + \
                                 spher2cart(r=target_dist,
                                            azimuth=target_orientation[0],
                                            colatitude=target_orientation[1]
                                            )

                    # make sure inside room and meets constraint
                    if not is_inside(target_loc, room_dim,
                                     self.source_min_dist_wall):
                        count += 1
                        continue

                    # sample noise
                    n_noise = self.n_noise_distrib.sample()
                    noise_loc = []
                    noise_orientations = []
                    noise_dists = []
                    for _ in range(n_noise):
                        noise_orientation = \
                            self.noise_orientation_distrib.sample()
                        noise_orientations.append(noise_orientation)
                        noise_dist = sample_source_distance(room, mic_loc,
                                                            noise_orientation)
                        if noise_dist < self.source_min_dist_mic:
                            break
                        noise_dists.append(noise_dist)
                        _noise_loc = mic_loc + \
                                     spher2cart(
                                         r=noise_dist,
                                         azimuth=noise_orientation[0],
                                         colatitude=noise_orientation[1])

                        # make sure inside room and meets constraint
                        if not is_inside(_noise_loc, room_dim,
                                         self.source_min_dist_wall):
                            break

                        noise_loc.append(_noise_loc.tolist())

                    if len(noise_loc) != n_noise:
                        # couldn't find good noise location(s)
                        count += 1
                        continue

                    # found valid configuration
                    break

                except NoValidRoom:
                    # try again
                    count += 1

            if count == self.timeout:
                print('Could not find valid room configuration. '
                      'One less room...')
                continue

            # gather room metadata to save
            room_uuid = 'room_' + str(uuid.uuid4())
            room_params = {
                'file': room_uuid + '.npz',
                'dimensions': room_dim,
                'mic_location': mic_loc,
                'target_location': target_loc.tolist(),
                'target_dist': int(target_dist),
                'target_orientation': target_orientation,
                'rt60': rt60,
                'absorption': energy_absorp,
                'n_noise': int(n_noise),
                'noise_loc': noise_loc,
                'noise_dist': noise_dists,
                'noise_orientation': noise_orientations,
            }
            room_metadata[room_uuid] = room_params

            from pprint import pprint
            pprint(room_params)

            # compute room impulse responses (RIRs)
            room.add_source(list(target_loc))
            for n in range(n_noise):
                room.add_source(list(noise_loc[n]))
            room.compute_rir()

            # collect RIRs
            n_mics = R.shape[1]
            target_ir = np.array([room.rir[n][0] for n in range(n_mics)])
            noise_irs = []
            for t in range(n_noise):
                noise_irs.append(
                    np.array([room.rir[n][t + 1] for n in range(n_mics)]))

            # save RIRs
            _output_file = os.path.join(data_dir, room_uuid + '.npz')
            if n_noise:
                noise_irs_dict = dict(
                    ('noise_ir_{}'.format(idx), ir)
                    for (idx, ir) in enumerate(noise_irs)
                )
                np.savez(
                    _output_file,
                    target_ir=target_ir,
                    sample_rate=self.sample_rate,
                    n_noise=n_noise,
                    **noise_irs_dict
                )
            else:
                np.savez(
                    _output_file,
                    target_ir=target_ir,
                    sample_rate=self.sample_rate,
                    n_noise=n_noise
                )

        # write metadata to JSON
        output_json = os.path.join(output_dir, 'room_metadata.json')
        with open(output_json, 'w') as f:
            json.dump(room_metadata, f, indent=4)

        print('Done.')


def rt60_to_absorption_eyring(room_dim, rt60):
    """

    Determine absorption factor given dimensions of (shoebox) room and RT60
    using Eyring's empirical equation.

    Parameters
    ------------
    room_dim : tuple / list
        Tuple / list of three elements, (width, length, height).
    rt60 : float
        Reverberation time (for 60 dB drop) in seconds.
    """
    if rt60 < 1e-5:
        return 1.
    else:
        width, length, height = room_dim
        vol = width * length * height
        area = 2 * (width * length + length * height + width * height)
        return 1. - np.exp(-0.16 * vol / rt60 / area)


class NoValidRoom(Exception):
    pass


def sample_mic_location(room_dim, mic_min_dist_wall):
    """

    Sample (x,y) coordinates of microphone location.

    Parameters
    ------------
    room_dim : tuple / list
        Tuple / list of three elements, (width, length, height)
    mic_min_dist_wall : float
        Minimum distance of mic from wall in meters.

    """

    width, length, _ = room_dim
    assert width >= 2 * mic_min_dist_wall and length >= 2 * mic_min_dist_wall
    return [
        np.random.uniform(mic_min_dist_wall, width - mic_min_dist_wall),
        np.random.uniform(mic_min_dist_wall, length - mic_min_dist_wall)
    ]


def is_inside(source_loc, room_dim, source_min_dist_wall):
    """

    Determine if source is inside room and meets the minimum distance to wall
    constraint. Problem is very simplified as we assume ShoeBox room with
    walls aligned to x, y, and z axis.

    Parameters
    ------------
    source_loc : 3D array
        x, y, and z coordinates of source in question.
    room_dim : 3D array
        width, length, and height of room
    source_min_dist_wall : float
        minimum distance from wall in meters.

    """
    for s, r in zip(*[source_loc, room_dim]):
        if s > r or s < source_min_dist_wall:
            return False
    return True


def sample_source_distance(room, mic_loc, orientation):
    """

    Sample a valid source distance.

    Parameters
    -----------
    room : Shoebox Room object
    mic_loc : array
        3D coordinates of microphone in room.
    orientation : array
        2D array containing azimuth and elevation angle of source.

    """

    # for shoebox, max possible distance is diagonal between extreme corners
    diag_dist = np.sqrt(sum(room.shoebox_dim ** 2))
    test_point = mic_loc + \
                 spher2cart(diag_dist, orientation[0], orientation[1])

    # determine intersection and then sample distance in between
    intersection = np.zeros(3, dtype=np.float32)
    mic_loc = np.array(mic_loc).astype(np.float32)
    test_point = np.array(test_point).astype(np.float32)
    for k, w in enumerate(room.walls):
        if w.intersection(mic_loc, test_point, intersection) == 0:
            max_dist = np.sqrt(sum((mic_loc - intersection) ** 2))
            return np.random.uniform(0, max_dist)
