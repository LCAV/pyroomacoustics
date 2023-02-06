# Shoebox is a sub-class of rooms that are rectangular
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
import numpy as np

from ..parameters import Material
from .room import Room


class ShoeBox(Room):
    """
    This class provides an API for creating a ShoeBox room in 2D or 3D.

    Parameters
    ----------
    p : array_like
        Length 2 (width, length) or 3 (width, length, height) depending on
        the desired dimension of the room.
    fs: int, optional
        The sampling frequency in Hz. Default is 8000.
    t0: float, optional
        The global starting time of the simulation in seconds. Default is 0.
    absorption : float
        Average amplitude absorption of walls. Note that this parameter is
        deprecated; use `materials` instead!
    max_order: int, optional
        The maximum reflection order in the image source model. Default is 1,
        namely direct sound and first order reflections.
    sigma2_awgn: float, optional
        The variance of the additive white Gaussian noise added during
        simulation. By default, none is added.
    sources: list of SoundSource objects, optional
        Sources to place in the room. Sources can be added after room creating
        with the `add_source` method by providing coordinates.
    mics: MicrophoneArray object, optional
        The microphone array to place in the room. A single microphone or
        microphone array can be added after room creation with the
        `add_microphone_array` method.
    materials : `Material` object or `dict` of `Material` objects
        See `pyroomacoustics.parameters.Material`. If providing a `dict`,
        you must provide a `Material` object for each wall: 'east',
        'west', 'north', 'south', 'ceiling' (3D), 'floor' (3D).
    temperature: float, optional
        The air temperature in the room in degree Celsius. By default, set so
        that speed of sound is 343 m/s.
    humidity: float, optional
        The relative humidity of the air in the room (between 0 and 100). By
        default set to 0.
    air_absorption: bool, optional
        If set to True, absorption of sound energy by the air will be
        simulated.
    ray_tracing: bool, optional
        If set to True, the ray tracing simulator will be used along with
        image source model.
    use_rand_ism: bool, optional
        If set to True, image source positions will have a small random
        displacement to prevent sweeping echoes
    max_rand_disp: float, optional;
        If using randomized image source method, what is the maximum
        displacement of the image sources?
    """

    def __init__(
        self,
        p,
        fs=8000,
        t0=0.0,
        absorption=None,  # deprecated
        max_order=1,
        sigma2_awgn=None,
        sources=None,
        mics=None,
        materials=None,
        temperature=None,
        humidity=None,
        air_absorption=False,
        ray_tracing=False,
        use_rand_ism=False,
        max_rand_disp=0.08,
        min_phase=False,
    ):
        p = np.array(p, dtype=np.float32)

        if len(p.shape) > 1 and (len(p) != 2 or len(p) != 3):
            raise ValueError("`p` must be a vector of length 2 or 3.")

        self.dim = p.shape[0]

        # record shoebox dimension in object
        self.shoebox_dim = np.array(p)

        # initialize the attributes of the room
        self._var_init(
            fs,
            t0,
            max_order,
            sigma2_awgn,
            temperature,
            humidity,
            air_absorption,
            ray_tracing,
            use_rand_ism,
            max_rand_disp,
            min_phase,
        )

        # Keep the correctly ordered naming of walls
        # This is the correct order for the shoebox computation later
        # W/E is for axis x, S/N for y-axis, F/C for z-axis
        self.wall_names = ["west", "east", "south", "north"]
        if self.dim == 3:
            self.wall_names += ["floor", "ceiling"]

        n_walls = len(self.wall_names)

        ############################
        # BEGIN COMPATIBILITY CODE #
        ############################

        if absorption is None:
            absorption_compatibility_request = False
            absorption = 0.0
        else:
            absorption_compatibility_request = True

        # copy over the absorption coefficient
        if isinstance(absorption, float):
            absorption = dict(zip(self.wall_names, [absorption] * n_walls))

        ##########################
        # END COMPATIBILITY CODE #
        ##########################

        if materials is not None:
            if absorption_compatibility_request:
                warnings.warn(
                    "Because `materials` were specified, deprecated "
                    "`absorption` parameter is ignored.",
                    DeprecationWarning,
                )

            if isinstance(materials, Material):
                materials = dict(zip(self.wall_names, [materials] * n_walls))
            elif not isinstance(materials, dict):
                raise ValueError(
                    "`materials` must be a `Material` object or "
                    "a `dict` specifying a `Material` object for"
                    " each wall: 'east', 'west', 'north', "
                    "'south', 'ceiling' (3D), 'floor' (3D)."
                )

            for w_name in self.wall_names:
                assert isinstance(
                    materials[w_name], Material
                ), "Material not specified using correct class"

        elif absorption_compatibility_request:
            warnings.warn(
                "Using absorption parameter is deprecated. Use `materials` with "
                "`Material` object instead.",
                DeprecationWarning,
            )

            # order the wall absorptions
            if not isinstance(absorption, dict):
                raise ValueError(
                    "`absorption` must be either a scalar or a "
                    "2x dim dictionary with entries for each "
                    "wall, namely: 'east', 'west', 'north', "
                    "'south', 'ceiling' (3d), 'floor' (3d)."
                )

            materials = {}
            for w_name in self.wall_names:
                if w_name in absorption:
                    # Fix the absorption
                    # 1 - a1 == sqrt(1 - a2)    <-- a1 is former incorrect absorption, a2 is the correct definition based on energy
                    # <=> a2 == 1 - (1 - a1) ** 2
                    correct_abs = 1.0 - (1.0 - absorption[w_name]) ** 2
                    materials[w_name] = Material(energy_absorption=correct_abs)
                else:
                    raise KeyError(
                        "Absorption needs to have keys 'east', 'west', "
                        "'north', 'south', 'ceiling' (3d), 'floor' (3d)."
                    )
        else:
            # In this case, no material is provided, use totally reflective
            # walls, no scattering
            materials = dict(
                zip(self.wall_names, [Material(energy_absorption=0.0)] * n_walls)
            )

        # If some of the materials used are multi-band, we need to resample
        # all of them to have the same number of values
        if not Material.all_flat(materials):
            for name, mat in materials.items():
                mat.resample(self.octave_bands)

        # Get the absorption and scattering as arrays
        # shape: (n_bands, n_walls)
        absorption_array = np.array(
            [materials[w].absorption_coeffs for w in self.wall_names]
        ).T
        scattering_array = np.array(
            [materials[w].scattering_coeffs for w in self.wall_names]
        ).T

        # Create the real room object
        self._init_room_engine(self.shoebox_dim, absorption_array, scattering_array)

        self.walls = self.room_engine.walls

        Room._wall_mapping(self)

        # add the sources
        self.sources = []
        if sources is not None and isinstance(sources, list):
            for src in sources:
                self.add_soundsource(src)

        # add the microphone array
        if mics is not None:
            self.add_microphone_array(mics)
        else:
            self.mic_array = None

    def extrude(self, height):
        """Overload the extrude method from 3D rooms"""

        if height < 0.0:
            raise ValueError("Room height must be positive")

        Room.extrude(self, np.array([0.0, 0.0, height]))

        # update the shoebox dim
        self.shoebox_dim = np.append(self.shoebox_dim, height)

    def get_volume(self):
        """
        Computes the volume of a room

        Returns
        -------
        the volume in cubic unit
        """

        return np.prod(self.shoebox_dim)

    def is_inside(self, pos):
        """
        Parameters
        ----------
        pos: array_like
            The position to test in an array of size 2 for a 2D room and 3 for a 3D room

        Returns
        -------
        True if ``pos`` is a point in the room, ``False`` otherwise.
        """
        pos = np.array(pos)
        return np.all(pos >= 0) and np.all(pos <= self.shoebox_dim)
