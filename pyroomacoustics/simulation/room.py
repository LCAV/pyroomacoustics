# Main Room class using to encapsulate the room acoustics simulator
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


import math
import warnings

import numpy as np
from scipy.interpolate import interp1d

from .. import beamforming as bf
from .. import libroom
from ..acoustics import OctaveBandsFactory, rt60_eyring, rt60_sabine
from ..beamforming import MicrophoneArray
from ..directivities import CardioidFamily, DIRPATRir, source_angle_shoebox
from ..experimental import measure_rt60
from ..libroom import Wall, Wall2D
from ..parameters import Material, Physics, constants, eps, make_materials
from ..soundsource import SoundSource
from ..utilities import angle_function
from .helpers import find_non_convex_walls, sequence_generation, wall_factory


class Room(object):
    """
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

    The Room is sub-classed by :py:obj:`pyroomacoustics.room.ShoeBox` which
    creates a rectangular (2D) or parallelepipedic (3D) room. Such rooms
    benefit from an efficient algorithm for the image source method.


    :attribute walls: (Wall array) list of walls forming the room
    :attribute fs: (int) sampling frequency
    :attribute max_order: (int) the maximum computed order for images
    :attribute sources: (SoundSource array) list of sound sources
    :attribute mics: (MicrophoneArray) array of microphones
    :attribute corners: (numpy.ndarray 2xN or 3xN, N=number of walls) array containing a point belonging to each wall, used for calculations
    :attribute absorption: (numpy.ndarray size N, N=number of walls)  array containing the absorption factor for each wall, used for calculations
    :attribute dim: (int) dimension of the room (2 or 3 meaning 2D or 3D)
    :attribute wallsId: (int dictionary) stores the mapping "wall name -> wall id (in the array walls)"

    Parameters
    ----------
    walls: list of Wall or Wall2D objects
        The walls forming the room.
    fs: int, optional
        The sampling frequency in Hz. Default is 8000.
    t0: float, optional
        The global starting time of the simulation in seconds. Default is 0.
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
    min_phase: bool, optional
        If set to True, generated RIRs will have a minimum phase response.
        Cannot be used with ray tracing model.
    """

    def __init__(
        self,
        walls,
        fs=8000,
        t0=0.0,
        max_order=1,
        sigma2_awgn=None,
        sources=None,
        mics=None,
        temperature=None,
        humidity=None,
        air_absorption=False,
        ray_tracing=False,
        use_rand_ism=False,
        max_rand_disp=0.08,
        min_phase=False,
    ):
        self.walls = walls

        # Get the room dimension from that of the walls
        self.dim = walls[0].dim

        # Create a mapping with friendly names for walls
        self._wall_mapping()

        # initialize everything else
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

        # initialize the C++ room engine
        self._init_room_engine()

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

    def _var_init(
        self,
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
    ):
        self.fs = fs

        if t0 != 0.0:
            raise NotImplementedError(
                "Global simulation delay not " "implemented (aka t0)"
            )
        self.t0 = t0

        self.max_order = max_order
        self.sigma2_awgn = sigma2_awgn

        self.octave_bands = OctaveBandsFactory(fs=self.fs)
        self.max_rand_disp = max_rand_disp

        # Keep track of the state of the simulator
        self.simulator_state = {
            "ism_needed": (self.max_order >= 0),
            "random_ism_needed": use_rand_ism,
            "rt_needed": ray_tracing,
            "air_abs_needed": air_absorption,
            "ism_done": False,
            "rt_done": False,
            "rir_done": False,
        }

        # make it clear the room (C++) engine is not ready yet
        self.room_engine = None

        if temperature is None and humidity is None:
            # default to package wide setting when nothing is provided
            self.physics = Physics().from_speed(constants.get("c"))
        else:
            # use formulas when temperature and/or humidity are provided
            self.physics = Physics(temperature=temperature, humidity=humidity)

        self.set_sound_speed(self.physics.get_sound_speed())
        self.air_absorption = None
        if air_absorption:
            self.set_air_absorption()

        # default values for ray tracing parameters
        self._set_ray_tracing_options(use_ray_tracing=ray_tracing)

        # in the beginning, nothing has been
        self.visibility = None

        # initialize the attribute for the impulse responses
        self.rir = None

        # self.sh_deg = 12
        # self.print_filter = 0

        self.min_phase = min_phase

    def _init_room_engine(self, *args):
        args = list(args)

        if len(args) == 0:
            # This is a polygonal room
            # find the non convex walls
            obstructing_walls = find_non_convex_walls(self.walls)
            args += [self.walls, obstructing_walls]

        # for shoebox rooms, the required arguments are passed to
        # the function

        # initialize the C++ room engine
        args += [
            [],
            self.c,  # speed of sound
            self.max_order,
            self.rt_args["energy_thres"],
            self.rt_args["time_thres"],
            self.rt_args["receiver_radius"],
            self.rt_args["hist_bin_size"],
            self.simulator_state["ism_needed"] and self.simulator_state["rt_needed"],
        ]

        # Create the real room object
        if self.dim == 2:
            self.room_engine = libroom.Room2D(*args)
        else:
            self.room_engine = libroom.Room(*args)

    def _update_room_engine_params(self):
        # Now, if it exists, set the parameters of room engine
        if self.room_engine is not None:
            self.room_engine.set_params(
                self.c,  # speed of sound
                self.max_order,
                self.rt_args["energy_thres"],
                self.rt_args["time_thres"],
                self.rt_args["receiver_radius"],
                self.rt_args["hist_bin_size"],
                (
                    self.simulator_state["ism_needed"]
                    and self.simulator_state["rt_needed"]
                ),
            )

    @property
    def is_multi_band(self):
        multi_band = False
        for w in self.walls:
            if len(w.absorption) > 1:
                multi_band = True
        return multi_band

    def set_ray_tracing(
        self,
        n_rays=None,
        receiver_radius=0.5,
        energy_thres=1e-7,
        time_thres=10.0,
        hist_bin_size=0.004,
    ):
        """
        Activates the ray tracer.

        Parameters
        ----------
        n_rays: int, optional
            The number of rays to shoot in the simulation
        receiver_radius: float, optional
            The radius of the sphere around the microphone in which to
            integrate the energy (default: 0.5 m)
        energy_thres: float, optional
            The energy thresold at which rays are stopped (default: 1e-7)
        time_thres: float, optional
            The maximum time of flight of rays (default: 10 s)
        hist_bin_size: float
            The time granularity of bins in the energy histogram (default: 4 ms)
        """
        self._set_ray_tracing_options(
            use_ray_tracing=True,
            n_rays=n_rays,
            receiver_radius=receiver_radius,
            energy_thres=energy_thres,
            time_thres=time_thres,
            hist_bin_size=hist_bin_size,
        )

    def _set_ray_tracing_options(
        self,
        use_ray_tracing,
        n_rays=None,
        receiver_radius=0.5,
        energy_thres=1e-7,
        time_thres=10.0,
        hist_bin_size=0.004,
        is_init=False,
    ):
        """
        Base method to set all ray tracing related options
        """

        if use_ray_tracing:
            if hasattr(self, "mic_array") and self.mic_array is not None:
                if self.mic_array.directivity is not None:
                    raise NotImplementedError(
                        "Directivity not supported with ray tracing."
                    )
            if hasattr(self, "sources"):
                for source in self.sources:
                    if source.directivity is not None:
                        raise NotImplementedError(
                            "Directivity not supported with ray tracing."
                        )

        self.simulator_state["rt_needed"] = use_ray_tracing

        self.rt_args = {}
        self.rt_args["energy_thres"] = energy_thres
        self.rt_args["time_thres"] = time_thres
        self.rt_args["receiver_radius"] = receiver_radius
        self.rt_args["hist_bin_size"] = hist_bin_size

        # set the histogram bin size so that it is an integer number of samples
        self.rt_args["hist_bin_size_samples"] = math.floor(
            self.fs * self.rt_args["hist_bin_size"]
        )
        self.rt_args["hist_bin_size"] = self.rt_args["hist_bin_size_samples"] / self.fs

        if n_rays is None:
            n_rays_auto_flag = True

            # We follow Vorlaender 2008, Eq. (11.12) to set the default number of rays
            # It depends on the mean hit rate we want to target
            target_mean_hit_count = 20

            # This is the multiplier for a single hit in average
            k1 = self.get_volume() / (
                np.pi
                * (self.rt_args["receiver_radius"] ** 2)
                * self.c
                * self.rt_args["hist_bin_size"]
            )

            n_rays = int(target_mean_hit_count * k1)

            if self.simulator_state["rt_needed"] and n_rays > 100000:
                import warnings

                warnings.warn(
                    "The number of rays used for ray tracing is larger than"
                    "100000 which may result in slow simulation.  The number"
                    "of rays was automatically chosen to provide accurate"
                    "room impulse response based on the room volume and the"
                    "receiver radius around the microphones.  The number of"
                    "rays may be reduced by increasing the size of the"
                    "receiver.  This tends to happen especially for large"
                    "rooms with small receivers.  The receiver is a sphere"
                    "around the microphone and its radius (in meters) may be"
                    "specified by providing the `receiver_radius` keyword"
                    "argument to the `set_ray_tracing` method."
                )

        self.rt_args["n_rays"] = n_rays

        self._update_room_engine_params()

    def unset_ray_tracing(self):
        """Deactivates the ray tracer"""
        self.simulator_state["rt_needed"] = False
        self._update_room_engine_params()

    def set_air_absorption(self, coefficients=None):
        """
        Activates or deactivates air absorption in the simulation.

        Parameters
        ----------
        coefficients: list of float
            List of air absorption coefficients, one per octave band
        """

        self.simulator_state["air_abs_needed"] = True
        if coefficients is None:
            self.air_absorption = self.octave_bands(**self.physics.get_air_absorption())
        else:
            # ignore temperature and humidity if coefficients are provided
            self.air_absorption = self.physics().get_air_absorption()

    def unset_air_absorption(self):
        """Deactivates air absorption in the simulation"""
        self.simulator_state["air_abs_needed"] = False

    def set_sound_speed(self, c):
        """Sets the speed of sound unconditionnaly"""
        self.c = c
        self._update_room_engine_params()

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
        fs=8000,
        t0=0.0,
        max_order=1,
        sigma2_awgn=None,
        sources=None,
        mics=None,
        materials=None,
        **kwargs,
    ):
        """
        Creates a 2D room by giving an array of corners.

        Parameters
        ----------
        corners: (np.array dim 2xN, N>2)
            list of corners, must be antiClockwise oriented
        absorption: float array or float
            list of absorption factor for each wall or single value
            for all walls (deprecated, use ``materials`` instead)
        fs: int, optional
            The sampling frequency in Hz. Default is 8000.
        t0: float, optional
            The global starting time of the simulation in seconds. Default is 0.
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
        kwargs: key, value mappings
            Other keyword arguments accepted by the :py:class:`~pyroomacoustics.room.Room` class

        Returns
        -------
        Instance of a 2D room
        """
        # make sure the corners are wrapped in an ndarray
        corners = np.array(corners)
        n_walls = corners.shape[1]

        corners = np.array(corners)
        if corners.shape[0] != 2 or n_walls < 3:
            raise ValueError("Arg corners must be more than two 2D points.")

        # We want to make sure the corners are ordered counter-clockwise
        if libroom.area_2d_polygon(corners) <= 0:
            corners = corners[:, ::-1]

        ############################
        # BEGIN COMPATIBILITY CODE #
        ############################

        if absorption is None:
            absorption = 0.0
            absorption_compatibility_request = False
        else:
            absorption_compatibility_request = True

        absorption = np.array(absorption, dtype="float64")
        if absorption.ndim == 0:
            absorption = absorption * np.ones(n_walls)
        elif absorption.ndim >= 1 and n_walls != len(absorption):
            raise ValueError(
                "Arg absorption must be the same size as corners or must be a single value."
            )

        ############################
        # BEGIN COMPATIBILITY CODE #
        ############################

        if materials is not None:
            if absorption_compatibility_request:
                import warnings

                warnings.warn(
                    "Because materials were specified, deprecated absorption parameter is ignored.",
                    DeprecationWarning,
                )

            if not isinstance(materials, list):
                materials = [materials] * n_walls

            if len(materials) != n_walls:
                raise ValueError("One material per wall is necessary.")

            for i in range(n_walls):
                assert isinstance(
                    materials[i], Material
                ), "Material not specified using correct class"

        elif absorption_compatibility_request:
            import warnings

            warnings.warn(
                "Using absorption parameter is deprecated. In the future, use materials instead."
            )

            # Fix the absorption
            # 1 - a1 == sqrt(1 - a2)    <-- a1 is former incorrect absorption, a2 is the correct definition based on energy
            # <=> a2 == 1 - (1 - a1) ** 2
            correct_absorption = 1.0 - (1.0 - absorption) ** 2
            materials = make_materials(*correct_absorption)

        else:
            # In this case, no material is provided, use totally reflective walls, no scattering
            materials = [Material(0.0, 0.0)] * n_walls

        # Resample material properties at octave bands
        octave_bands = OctaveBandsFactory(fs=fs)
        if not Material.all_flat(materials):
            for mat in materials:
                mat.resample(octave_bands)

        # Create the walls
        walls = []
        for i in range(n_walls):
            walls.append(
                wall_factory(
                    np.array([corners[:, i], corners[:, (i + 1) % n_walls]]).T,
                    materials[i].absorption_coeffs,
                    materials[i].scattering_coeffs,
                    "wall_" + str(i),
                )
            )

        return cls(
            walls,
            fs=fs,
            t0=t0,
            max_order=max_order,
            sigma2_awgn=sigma2_awgn,
            sources=sources,
            mics=mics,
            **kwargs,
        )

    def extrude(self, height, v_vec=None, absorption=None, materials=None):
        """
        Creates a 3D room by extruding a 2D polygon.
        The polygon is typically the floor of the room and will have z-coordinate zero. The ceiling

        Parameters
        ----------
        height : float
            The extrusion height
        v_vec : array-like 1D length 3, optional
            A unit vector. An orientation for the extrusion direction. The
            ceiling will be placed as a translation of the floor with respect
            to this vector (The default is [0,0,1]).
        absorption : float or array-like, optional
            Absorption coefficients for all the walls. If a scalar, then all the walls
            will have the same absorption. If an array is given, it should have as many elements
            as there will be walls, that is the number of vertices of the polygon plus two. The two
            last elements are for the floor and the ceiling, respectively.
            It is recommended to use materials instead of absorption parameter. (Default: 1)
        materials : dict
            Absorption coefficients for floor and ceiling. This parameter overrides absorption.
            (Default: {"floor": 1, "ceiling": 1})
        """

        if self.dim != 2:
            raise ValueError("Can only extrude a 2D room.")

        # default orientation vector is pointing up
        if v_vec is None:
            v_vec = np.array([0.0, 0.0, 1.0])

        # check that the walls are ordered counterclock wise
        # that should be the case if created from from_corners function
        nw = len(self.walls)
        floor_corners = np.zeros((2, nw))
        floor_corners[:, 0] = self.walls[0].corners[:, 0]
        ordered = True
        for iw, wall in enumerate(self.walls[1:]):
            if not np.allclose(self.walls[iw].corners[:, 1], wall.corners[:, 0]):
                ordered = False
            floor_corners[:, iw + 1] = wall.corners[:, 0]
        if not np.allclose(self.walls[-1].corners[:, 1], self.walls[0].corners[:, 0]):
            ordered = False

        if not ordered:
            raise ValueError(
                "The wall list should be ordered counter-clockwise, which is the case \
                if the room is created with Room.from_corners"
            )

        # make sure the floor_corners are ordered anti-clockwise (for now)
        if libroom.area_2d_polygon(floor_corners) <= 0:
            floor_corners = np.fliplr(floor_corners)

        walls = []
        for i in range(nw):
            corners = np.array(
                [
                    np.r_[floor_corners[:, i], 0],
                    np.r_[floor_corners[:, (i + 1) % nw], 0],
                    np.r_[floor_corners[:, (i + 1) % nw], 0] + height * v_vec,
                    np.r_[floor_corners[:, i], 0] + height * v_vec,
                ]
            ).T
            walls.append(
                wall_factory(
                    corners,
                    self.walls[i].absorption,
                    self.walls[i].scatter,
                    name=str(i),
                )
            )

        ############################
        # BEGIN COMPATIBILITY CODE #
        ############################
        if absorption is not None:
            absorption = 0.0
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
                assert isinstance(
                    mat, Material
                ), "Material not specified using correct class"

        elif absorption_compatibility_request:
            import warnings

            warnings.warn(
                "absorption parameter is deprecated for Room.extrude",
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

            materials = make_materials(
                floor=(absorption[0], 0.0), ceiling=(absorption[0], 0.0)
            )

        else:
            # In this case, no material is provided, use totally reflective walls, no scattering
            new_mat = Material(0.0, 0.0)
            materials = {"floor": new_mat, "ceiling": new_mat}

        new_corners = {}
        new_corners["floor"] = np.pad(floor_corners, ((0, 1), (0, 0)), mode="constant")
        new_corners["ceiling"] = (new_corners["floor"].T + height * v_vec).T

        # we need the floor corners to ordered clockwise (for the normal to point outward)
        new_corners["floor"] = np.fliplr(new_corners["floor"])

        for key in ["floor", "ceiling"]:
            walls.append(
                wall_factory(
                    new_corners[key],
                    materials[key].absorption_coeffs,
                    materials[key].scattering_coeffs,
                    name=key,
                )
            )

        self.walls = walls
        self.dim = 3

        # Update the real room object
        self._init_room_engine()

    def plot(
        self,
        img_order=None,
        freq=None,
        figsize=None,
        no_axis=False,
        mic_marker_size=10,
        plot_directivity=True,
        ax=None,
        **kwargs,
    ):
        """Plots the room with its walls, microphones, sources and images"""

        try:
            import matplotlib
            import matplotlib.pyplot as plt
            from matplotlib.collections import PatchCollection
            from matplotlib.patches import Circle, Polygon, Wedge
        except ImportError:
            import warnings

            warnings.warn("Matplotlib is required for plotting")
            return

        fig = None

        if self.dim == 2:
            fig = plt.figure(figsize=figsize)

            if no_axis is True:
                if ax is None:
                    ax = fig.add_axes([0, 0, 1, 1], aspect="equal", **kwargs)
                ax.axis("off")
                rect = fig.patch
                rect.set_facecolor("gray")
                rect.set_alpha(0.15)
            else:
                if ax is None:
                    ax = fig.add_subplot(111, aspect="equal", **kwargs)

            # draw room
            corners = np.array([wall.corners[:, 0] for wall in self.walls]).T
            polygons = [Polygon(xy=corners.T, closed=True)]
            p = PatchCollection(
                polygons,
                cmap=matplotlib.cm.jet,
                facecolor=np.array([1, 1, 1]),
                edgecolor=np.array([0, 0, 0]),
            )
            ax.add_collection(p)

            if self.mic_array is not None:
                for i in range(self.mic_array.nmic):
                    ax.scatter(
                        self.mic_array.R[0][i],
                        self.mic_array.R[1][i],
                        marker="x",
                        linewidth=0.5,
                        s=mic_marker_size,
                        c="k",
                    )

                    if plot_directivity and self.mic_array.directivity is not None:
                        azimuth_plot = np.linspace(
                            start=0, stop=360, num=361, endpoint=True
                        )
                        ax = self.mic_array.directivity[i].plot_response(
                            azimuth=azimuth_plot,
                            degrees=True,
                            ax=ax,
                            offset=self.mic_array.R[:, i],
                        )

                # draw the beam pattern of the beamformer if requested (and available)
                if (
                    freq is not None
                    and isinstance(self.mic_array, bf.Beamformer)
                    and (
                        self.mic_array.weights is not None
                        or self.mic_array.filters is not None
                    )
                ):
                    freq = np.array(freq)
                    if freq.ndim == 0:
                        freq = np.array([freq])

                    # define a new set of colors for the beam patterns
                    newmap = plt.get_cmap("autumn")
                    desat = 0.7
                    try:
                        # this is for matplotlib >= 2.0.0
                        ax.set_prop_cycle(
                            color=[
                                newmap(k) for k in desat * np.linspace(0, 1, len(freq))
                            ]
                        )
                    except:
                        # keep this for backward compatibility
                        ax.set_color_cycle(
                            [newmap(k) for k in desat * np.linspace(0, 1, len(freq))]
                        )

                    phis = np.arange(360) * 2 * np.pi / 360.0
                    newfreq = np.zeros(freq.shape)
                    H = np.zeros((len(freq), len(phis)), dtype=complex)
                    for i, f in enumerate(freq):
                        newfreq[i], H[i] = self.mic_array.response(phis, f)

                    # normalize max amplitude to one
                    H = np.abs(H) ** 2 / np.abs(H).max() ** 2

                    # a normalization factor according to room size
                    norm = np.linalg.norm(
                        (corners - self.mic_array.center), axis=0
                    ).max()

                    # plot all the beam patterns
                    for f, h in zip(newfreq, H):
                        x = np.cos(phis) * h * norm + self.mic_array.center[0, 0]
                        y = np.sin(phis) * h * norm + self.mic_array.center[1, 0]
                        ax.plot(x, y, "-", linewidth=0.5)

            # define some markers for different sources and colormap for damping
            markers = ["o", "s", "v", "."]
            cmap = plt.get_cmap("YlGnBu")

            # use this to check some image sources were drawn
            has_drawn_img = False

            # draw the scatter of images
            for i, source in enumerate(self.sources):
                # draw source
                ax.scatter(
                    source.position[0],
                    source.position[1],
                    c=[cmap(1.0)],
                    s=20,
                    marker=markers[i % len(markers)],
                    edgecolor=cmap(1.0),
                )

                if plot_directivity and source.directivity is not None:
                    azimuth_plot = np.linspace(
                        start=0, stop=360, num=361, endpoint=True
                    )
                    ax = source.directivity.plot_response(
                        azimuth=azimuth_plot,
                        degrees=True,
                        ax=ax,
                        offset=source.position,
                    )

                # draw images
                if img_order is None:
                    img_order = 0
                elif img_order == "max":
                    img_order = self.max_order

                I = source.orders <= img_order
                if len(I) > 0:
                    has_drawn_img = True

                val = (np.log2(np.mean(source.damping, axis=0)[I]) + 10.0) / 10.0
                # plot the images
                ax.scatter(
                    source.images[0, I],
                    source.images[1, I],
                    c=cmap(val),
                    s=20,
                    marker=markers[i % len(markers)],
                    edgecolor=cmap(val),
                )

            # When no image source has been drawn, we need to use the bounding box
            # to set correctly the limits of the plot
            if not has_drawn_img or img_order == 0:
                bbox = self.get_bbox()
                ax.set_xlim(bbox[0, :])
                ax.set_ylim(bbox[1, :])

            return fig, ax

        if self.dim == 3:
            import matplotlib.colors as colors
            import matplotlib.pyplot as plt
            import mpl_toolkits.mplot3d as a3

            if ax is None:
                fig = plt.figure(figsize=figsize)
                ax = a3.Axes3D(fig, auto_add_to_figure=False)
                fig.add_axes(ax)

            # plot the walls
            for w in self.walls:
                tri = a3.art3d.Poly3DCollection([w.corners.T], alpha=0.5)
                tri.set_color(colors.rgb2hex(np.random.rand(3)))
                tri.set_edgecolor("k")
                ax.add_collection3d(tri)

            # define some markers for different sources and colormap for damping
            markers = ["o", "s", "v", "."]
            cmap = plt.get_cmap("YlGnBu")

            # use this to check some image sources were drawn
            has_drawn_img = False

            # draw the scatter of images
            for i, source in enumerate(self.sources):
                # draw source
                ax.scatter(
                    source.position[0],
                    source.position[1],
                    source.position[2],
                    c=[cmap(1.0)],
                    s=20,
                    marker=markers[i % len(markers)],
                    edgecolor=cmap(1.0),
                )

                if plot_directivity and source.directivity is not None:
                    azimuth_plot = np.linspace(
                        start=0, stop=360, num=361, endpoint=True
                    )
                    colatitude_plot = np.linspace(
                        start=0, stop=180, num=180, endpoint=True
                    )
                    ax = source.directivity.plot_response(
                        azimuth=azimuth_plot,
                        colatitude=colatitude_plot,
                        degrees=True,
                        ax=ax,
                        offset=source.position,
                    )

                # draw images
                if img_order is None:
                    img_order = self.max_order

                I = source.orders <= img_order
                if len(I) > 0:
                    has_drawn_img = True

                val = (np.log2(np.mean(source.damping, axis=0)[I]) + 10.0) / 10.0
                # plot the images
                ax.scatter(
                    source.images[0, I],
                    source.images[1, I],
                    source.images[2, I],
                    c=cmap(val),
                    s=20,
                    marker=markers[i % len(markers)],
                    edgecolor=cmap(val),
                )

            # When no image source has been drawn, we need to use the bounding box
            # to set correctly the limits of the plot
            if not has_drawn_img or img_order == 0:
                bbox = self.get_bbox()
                ax.set_xlim3d(bbox[0, :])
                ax.set_ylim3d(bbox[1, :])
                ax.set_zlim3d(bbox[2, :])

            # draw the microphones
            if self.mic_array is not None:
                for i in range(self.mic_array.nmic):
                    ax.scatter(
                        self.mic_array.R[0][i],
                        self.mic_array.R[1][i],
                        self.mic_array.R[2][i],
                        marker="x",
                        linewidth=0.5,
                        s=mic_marker_size,
                        c="k",
                    )

                    if plot_directivity and self.mic_array.directivity is not None:
                        azimuth_plot = np.linspace(
                            start=0, stop=360, num=361, endpoint=True
                        )
                        colatitude_plot = np.linspace(
                            start=0, stop=180, num=180, endpoint=True
                        )
                        ax = self.mic_array.directivity[i].plot_response(
                            azimuth=azimuth_plot,
                            colatitude=colatitude_plot,
                            degrees=True,
                            ax=ax,
                            offset=self.mic_array.R[:, i],
                        )

            return fig, ax

    def plot_rir(self, select=None, FD=False, kind=None):
        """
        Plot room impulse responses. Compute if not done already.

        Parameters
        ----------
        select: list of tuples OR int
            List of RIR pairs `(mic, src)` to plot, e.g. `[(0,0), (0,1)]`. Or
            `int` to plot RIR from particular microphone to all sources. Note
            that microphones and sources are zero-indexed. Default is to plot
            all microphone-source pairs.
        FD: bool, optional
            If True, the transfer function is plotted instead of the impulse response.
            Default is False.
        kind: str, optional
            The value can be "ir", "tf", or "spec" which will plot impulse response,
            transfer function, and spectrogram, respectively. If this option is
            specified, then the value of ``FD`` is ignored. Default is "ir".


        Returns
        -------
        fig: matplotlib figure
            Figure object for further modifications
        axes: matplotlib list of axes objects
            Axes for further modifications
        """

        if kind is None:
            kind = "tf" if FD else "ir"

        if kind == "ir":
            y_label = None
            x_label = "Time (ms)"
        elif kind == "tf":
            x_label = "Freq. (kHz)"
            y_label = "Power (dB)"
        elif kind == "spec":
            x_label = "Time (ms)"
            y_label = "Freq. (kHz)"
        else:
            raise ValueError("The value of 'kind' should be 'ir', 'tf', or 'spec'.")

        n_src = len(self.sources)
        n_mic = self.mic_array.M
        if select is None:
            pairs = [(r, s) for r in range(n_mic) for s in range(n_src)]
        elif isinstance(select, int):
            pairs = [(select, s) for s in range(n_src)]
        elif isinstance(select, list) or isinstance(select, tuple):
            if (
                len(select) == 2
                and isinstance(select[0], int)
                and isinstance(select[1], int)
            ):
                pairs = [select]
            else:
                pairs = select
        else:
            raise ValueError('Invalid type for "select".')

        if not self.simulator_state["rir_done"]:
            self.compute_rir()

        # for plotting
        n_mic = len(list(set(pair[0] for pair in pairs)))
        n_src = len(list(set(pair[1] for pair in pairs)))
        r_plot = dict()
        s_plot = dict()
        for k, r in enumerate(list(set(pair[0] for pair in pairs))):
            r_plot[r] = k
        for k, s in enumerate(list(set(pair[1] for pair in pairs))):
            s_plot[s] = k

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            import warnings

            warnings.warn("Matplotlib is required for plotting")
            return

        def plot_func(ax, h):
            if kind == "ir":
                ax.plot(np.arange(len(h)) / float(self.fs / 1000), h)
            elif kind == "tf":
                H = 20.0 * np.log10(abs(np.fft.rfft(h)) + 1e-15)
                freq = np.arange(H.shape[0]) / h.shape[0] * (self.fs * 1000)
                ax.plot(freq, H)
            elif kind == "spec":
                h = h + np.random.randn(*h.shape) * 1e-15
                ax.specgram(h, Fs=self.fs / 1000)
            else:
                raise ValueError("The value of 'kind' should be 'ir', 'tf', or 'spec'.")

        if select is None:
            fig, axes = plt.subplots(
                n_mic, n_src, squeeze=False, sharex=True, sharey=True
            )
            for r in range(n_mic):
                for s in range(n_src):
                    h = self.rir[r][s]
                    plot_func(axes[r, s], h)

            for r in range(n_mic):
                if y_label is not None:
                    axes[r, 0].set_ylabel(y_label)

                axes[r, -1].annotate(
                    "Mic {}".format(r),
                    xy=(1.02, 0.5),
                    xycoords="axes fraction",
                    rotation=270,
                    ha="left",
                    va="center",
                )

            for s in range(n_src):
                axes[0, s].set_title("Source {}".format(s), fontsize="medium")
                if x_label is not None:
                    axes[-1, s].set_xlabel(x_label)

            fig.align_ylabels(axes[:, 0])
            fig.tight_layout()

        else:
            fig, axes = plt.subplots(
                len(pairs), 1, squeeze=False, sharex=True, sharey=True
            )
            for k, (r, s) in enumerate(pairs):
                h = self.rir[r][s]
                plot_func(axes[k, 0], h)

                if len(pairs) == 1:
                    axes[k, 0].set_title("Mic {}, Source {}".format(r, s))
                else:
                    axes[k, 0].annotate(
                        "M{}, S{}".format(r, s),
                        xy=(1.02, 0.5),
                        xycoords="axes fraction",
                        rotation=270,
                        ha="left",
                        va="center",
                    )

                if y_label is not None:
                    axes[k, 0].set_ylabel(y_label)

            if x_label is not None:
                axes[-1, 0].set_xlabel(x_label)
            fig.align_ylabels(axes[:, 0])
            fig.tight_layout()

        return fig, axes

    def add(self, obj):
        """
        Adds a sound source or microphone to a room

        Parameters
        ----------
        obj: :py:obj:`~pyroomacoustics.soundsource.SoundSource` or :py:obj:`~pyroomacoustics.beamforming.Microphone` object
            The object to add

        Returns
        -------
        :py:obj:`~pyroomacoustics.room.Room`
            The room is returned for further tweaking.
        """

        if isinstance(obj, SoundSource):
            if obj.dim != self.dim:
                raise ValueError(
                    (
                        "The Room and SoundSource objects must be of the same "
                        "dimensionality. The Room is {}D but the SoundSource "
                        "is {}D"
                    ).format(self.dim, obj.dim)
                )

            if not self.is_inside(np.array(obj.position)):
                raise ValueError("The source must be added inside the room.")

            self.sources.append(obj)

        elif isinstance(obj, MicrophoneArray):
            if obj.dim != self.dim:
                raise ValueError(
                    (
                        "The Room and MicrophoneArray objects must be of the same "
                        "dimensionality. The Room is {}D but the MicrophoneArray "
                        "is {}D"
                    ).format(self.dim, obj.dim)
                )

            if "mic_array" not in self.__dict__ or self.mic_array is None:
                self.mic_array = obj
            else:
                self.mic_array.append(obj)

            # microphone need to be added to the room_engine
            for m in range(len(obj)):
                self.room_engine.add_mic(obj.R[:, None, m])

        else:
            raise TypeError(
                "The add method from Room only takes SoundSource or "
                "MicrophoneArray objects as parameter"
            )

        return self

    def add_microphone(self, loc, fs=None, directivity=None):
        """
        Adds a single microphone in the room.

        Parameters
        ----------
        loc: array_like or ndarray
            The location of the microphone. The length should be the same as the room dimension.
        fs: float, optional
            The sampling frequency of the microphone, if different from that of the room.

        Returns
        -------
        :py:obj:`~pyroomacoustics.room.Room`
            The room is returned for further tweaking.
        """

        if self.simulator_state["rt_needed"] and directivity is not None:
            raise NotImplementedError("Directivity not supported with ray tracing.")

        # make sure this is a
        loc = np.array(loc)

        # if array, make it a 2D array as expected
        if loc.ndim == 1:
            loc = loc[:, None]

        if fs is None:
            fs = self.fs

        return self.add(MicrophoneArray(loc, fs, directivity))

    def add_microphone_array(self, mic_array, directivity=None):
        """
        Adds a microphone array (i.e. several microphones) in the room.

        Parameters
        ----------
        mic_array: array_like or ndarray or MicrophoneArray object
            The array can be provided as an array of size ``(dim, n_mics)``,
            where ``dim`` is the dimension of the room and ``n_mics`` is the
            number of microphones in the array.

            As an alternative, a
            :py:obj:`~pyroomacoustics.beamforming.MicrophoneArray` can be
            provided.

        Returns
        -------
        :py:obj:`~pyroomacoustics.room.Room`
            The room is returned for further tweaking.
        """

        if self.simulator_state["rt_needed"] and directivity is not None:
            raise NotImplementedError("Directivity not supported with ray tracing.")

        if not isinstance(mic_array, MicrophoneArray):
            # if the type is not a microphone array, try to parse a numpy array
            mic_array = MicrophoneArray(mic_array, self.fs, directivity)
        else:
            # if the type is microphone array
            if directivity is not None:
                mic_array.set_directivity(directivity)

            if self.simulator_state["rt_needed"] and mic_array.directivity is not None:
                raise NotImplementedError("Directivity not supported with ray tracing.")

        return self.add(mic_array)

    def add_source(self, position, signal=None, delay=0, directivity=None):
        """
        Adds a sound source given by its position in the room. Optionally
        a source signal and a delay can be provided.

        Parameters
        -----------
        position: ndarray, shape: (2,) or (3,)
            The location of the source in the room
        signal: ndarray, shape: (n_samples,), optional
            The signal played by the source
        delay: float, optional
            A time delay until the source signal starts
            in the simulation

        Returns
        -------
        :py:obj:`~pyroomacoustics.room.Room`
            The room is returned for further tweaking.
        """

        if self.simulator_state["rt_needed"] and directivity is not None:
            raise NotImplementedError("Directivity not supported with ray tracing.")

        if directivity is not None:
            from pyroomacoustics import ShoeBox

            if not isinstance(self, ShoeBox):
                raise NotImplementedError(
                    "Source directivity only supported for ShoeBox room."
                )

        if isinstance(position, SoundSource):
            if directivity is not None:
                if isinstance(directivity, CardioidFamily) or isinstance(
                    directivity, DIRPATRir
                ):
                    return self.add(SoundSource(position, directivity=directivity))
            else:
                return self.add(position)
        else:
            if directivity is not None:
                if isinstance(directivity, CardioidFamily) or isinstance(
                    directivity, DIRPATRir
                ):
                    return self.add(
                        SoundSource(
                            position,
                            signal=signal,
                            delay=delay,
                            directivity=directivity,
                        )
                    )

            else:
                return self.add(SoundSource(position, signal=signal, delay=delay))

    def add_soundsource(self, sndsrc, directivity=None):
        """
        Adds a :py:obj:`pyroomacoustics.soundsource.SoundSource` object to the room.

        Parameters
        ----------
        sndsrc: :py:obj:`~pyroomacoustics.soundsource.SoundSource` object
            The SoundSource object to add to the room
        """
        if directivity is not None:
            sndsrc.set_directivity(directivity)
        return self.add(sndsrc)

    def image_source_model(self):
        if not self.simulator_state["ism_needed"]:
            return

        self.visibility = []

        for source in self.sources:
            n_sources = self.room_engine.image_source_model(source.position)

            if n_sources > 0:  # Number of image source that are generated
                # Copy to python managed memory

                source.images = (
                    self.room_engine.sources.copy()
                )  # Positions of the image source (3,n) n: n_sources
                source.orders = (
                    self.room_engine.orders.copy()
                )  # Reflection order for each image source shape n:n_sources
                source.orders_xyz = (
                    self.room_engine.orders_xyz.copy()
                )  # Reflection order for each image source for each coordinate shape (3,n) n:n_sources
                source.walls = (
                    self.room_engine.gen_walls.copy()
                )  # Something that i don't get [-1,-1,-1,-1,-1...] shape n:n_sources
                source.damping = (
                    self.room_engine.attenuations.copy()
                )  # Octave band damping's shape (no_of_octave_bands*n_sources) damping value for each image source for each octave bands
                source.generators = -np.ones(source.walls.shape)

                # if randomized image method is selected, add a small random
                # displacement to the image sources

                if self.simulator_state["random_ism_needed"]:
                    n_images = np.shape(source.images)[1]

                    # maximum allowed displacement is 8cm
                    max_disp = self.max_rand_disp

                    # add a random displacement to each cartesian coordinate
                    disp = np.random.uniform(-max_disp, max_disp, size=(3, n_images))
                    source.images += disp

                self.visibility.append(self.room_engine.visible_mics.copy())

                # We need to check that microphones are indeed in the room
                for m in range(self.mic_array.R.shape[1]):
                    # if not, it's not visible from anywhere!
                    if not self.is_inside(self.mic_array.R[:, m]):
                        self.visibility[-1][m, :] = 0

        # Update the state
        self.simulator_state["ism_done"] = True

    def ray_tracing(self):
        if not self.simulator_state["rt_needed"]:
            return

        # this will be a list of lists with
        # shape (n_mics, n_src, n_directions, n_bands, n_time_bins)
        self.rt_histograms = [[] for r in range(self.mic_array.M)]

        for s, src in enumerate(self.sources):
            self.room_engine.ray_tracing(self.rt_args["n_rays"], src.position)

            for r in range(self.mic_array.M):
                self.rt_histograms[r].append([])
                for h in self.room_engine.microphones[r].histograms:
                    # get a copy of the histogram
                    self.rt_histograms[r][s].append(h.get_hist())
            # reset all the receivers' histograms
            self.room_engine.reset_mics()

        # Basically, histograms for 2 mics corresponding to each source , the histograms are in each octave bands hence (7,2500) 2500 histogram length
        # update the state
        self.simulator_state["rt_done"] = True

    def compute_rir(self):
        """
        Compute the room impulse response between every source and microphone.
        """

        if self.simulator_state["ism_needed"] and not self.simulator_state["ism_done"]:
            self.image_source_model()

        if self.simulator_state["rt_needed"] and not self.simulator_state["rt_done"]:
            self.ray_tracing()

        self.rir = []

        volume_room = self.get_volume()

        for m, mic in enumerate(
            self.mic_array.R.T
        ):  # Loop over ever microphone present in the room and then for each microphone and source pair present in the room
            self.rir.append([])
            for s, src in enumerate(self.sources):
                """
                Compute the room impulse response between the source
                and the microphone whose position is given as an
                argument.
                """
                # fractional delay length
                fdl = constants.get("frac_delay_length")

                fdl2 = fdl // 2

                # default, just in case both ism and rt are disabled (should never happen)
                N = fdl

                if self.simulator_state["ism_needed"]:
                    # compute azimuth and colatitude angles for receiver
                    if self.mic_array.directivity is not None:
                        angle_function_array = angle_function(src.images, mic)
                        azimuth_m = angle_function_array[0]
                        colatitude_m = angle_function_array[1]
                    else:
                        azimuth_m, colatitude_m = [], []

                    # compute azimuth and colatitude angles for source
                    if self.sources[s].directivity is not None:
                        azimuth_s, colatitude_s = source_angle_shoebox(
                            image_source_loc=src.images,
                            wall_flips=abs(src.orders_xyz),
                            mic_loc=mic,
                        )
                    else:
                        azimuth_s, colatitude_s = [], []

                    # compute the distance from image sources

                    dist = np.sqrt(
                        np.sum((src.images - mic[:, None]) ** 2, axis=0)
                    )  # Calculate distance between image sources and for each microphone

                    # dist shape (n) : n0 of image sources
                    time = (
                        dist / self.c
                    )  # Calculate time of arrival for each image source
                    t_max = (
                        time.max()
                    )  # The image source which takes the most time to arrive to this particular microphone
                    N = int(
                        math.ceil(t_max * self.fs)
                    )  # What will be the length of RIR according to t_max
                    print("Minimum Time", time.min() * self.fs)

                else:
                    t_max = 0.0

                if self.simulator_state["rt_needed"]:
                    # get the maximum length from the histograms
                    nz_bins_loc = np.nonzero(self.rt_histograms[m][s][0].sum(axis=0))[
                        0
                    ]  # Sum vertically across octave band for each value in histogram (7,2500) -> (2500) -> np .nonzero(

                    if len(nz_bins_loc) == 0:
                        n_bins = 0
                    else:
                        n_bins = nz_bins_loc[-1] + 1

                    t_max = np.maximum(t_max, n_bins * self.rt_args["hist_bin_size"])

                    # N changes here , the length of RIR changes if we apply RT method.
                    # the number of samples needed
                    # round up to multiple of the histogram bin size
                    # add the lengths of the fractional delay filter

                    hbss = int(self.rt_args["hist_bin_size_samples"])

                    N = int(math.ceil(t_max * self.fs / hbss) * hbss)

                # this is where we will compose the RIR i.e length of RIR
                ir = np.zeros(N + fdl)

                # This is the distance travelled wrt time
                distance_rir = np.arange(N) / self.fs * self.c

                # this is the random sequence for the tail generation
                seq = sequence_generation(volume_room, N / self.fs, self.c, self.fs)
                seq = seq[:N]  # take values according to N as seq is larger

                # Do band-wise RIR construction
                is_multi_band = self.is_multi_band
                bws = self.octave_bands.get_bw() if is_multi_band else [self.fs / 2]
                rir_bands = []

                """
                Use octave bands to construct RIR :
                1) Ray-tracing is activated
                2) directivity of both the microphones and source is not given.

                """

                if (
                    (
                        self.mic_array.directivity is None
                        or isinstance(self.mic_array.directivity[m], CardioidFamily)
                    )
                    and (
                        self.sources[s].directivity is None
                        or isinstance(self.sources[s].directivity, CardioidFamily)
                    )
                ) or self.simulator_state["rt_needed"]:
                    for b, bw in enumerate(bws):  # Loop through every band
                        ir_loc = np.zeros_like(ir)  # ir for every band

                        # IS method
                        if self.simulator_state["ism_needed"]:
                            alpha = (
                                src.damping[b, :] / dist
                            )  # calculate alpha according to every octave band and for all the image sources for this particular microphone

                            if self.mic_array.directivity is not None and isinstance(
                                self.mic_array.directivity[m], CardioidFamily
                            ):
                                alpha *= self.mic_array.directivity[m].get_response(
                                    azimuth=azimuth_m,
                                    colatitude=colatitude_m,
                                    frequency=bw,
                                    degrees=False,
                                )
                                print("Cmic")

                            if self.sources[s].directivity is not None and isinstance(
                                self.sources[s].directivity, CardioidFamily
                            ):
                                alpha *= self.sources[s].directivity.get_response(
                                    azimuth=azimuth_s,
                                    colatitude=colatitude_s,
                                    frequency=bw,
                                    degrees=False,
                                )
                                print("Csrc")

                            # Use the Cython extension for the fractional delays
                            from ..build_rir import fast_rir_builder

                            vis = self.visibility[s][m, :].astype(np.int32)

                            # we add the delay due to the factional delay filter to
                            # the arrival times to avoid problems when propagation
                            # is shorter than the delay to to the filter
                            # hence: time + fdl2

                            time_adjust = (
                                time + fdl2 / self.fs
                            )  # This remains the same for all octave bands.

                            fast_rir_builder(
                                ir_loc, time_adjust, alpha, vis, self.fs, fdl
                            )

                            if is_multi_band:
                                ir_loc = self.octave_bands.analysis(ir_loc, band=b)

                            ir += ir_loc  # All the IR'S from different octave bands are added together in the same sequence.

                        # Ray Tracing
                        if self.simulator_state["rt_needed"]:
                            if is_multi_band:
                                seq_bp = self.octave_bands.analysis(seq, band=b)

                            else:
                                seq_bp = seq.copy()

                            # interpolate the histogram and multiply the sequence

                            seq_bp_rot = seq_bp.reshape((-1, hbss))  # shape 72,64

                            new_n_bins = seq_bp_rot.shape[0]

                            # Take only those bins which have some non-zero values for that specific octave bands.

                            hist = self.rt_histograms[m][s][0][b, :new_n_bins]

                            normalization = np.linalg.norm(
                                seq_bp_rot, axis=1
                            )  # Take normalize of the poisson distribution octave band filtered array on the axis 1 -> shape (72|71) if input is of size (72,64)

                            # Only those indices which have normalization greater than 0.0
                            indices = normalization > 0.0

                            seq_bp_rot[indices, :] /= normalization[indices, None]

                            seq_bp_rot *= np.sqrt(hist[:, None])

                            # Normalize the band power
                            # The bands should normally sum up to fs / 2

                            seq_bp *= np.sqrt(bw / self.fs * 2.0)

                            ir_loc[fdl2 : fdl2 + N] += seq_bp

                        # keep for further processing

                        rir_bands.append(
                            ir_loc
                        )  # Impulse response for every octave band for each microphone

                    # Do Air absorption
                    if self.simulator_state["air_abs_needed"]:
                        # In case this was not multi-band, do the band pass filtering
                        if len(rir_bands) == 1:
                            rir_bands = self.octave_bands.analysis(rir_bands[0]).T

                        # Now apply air absorption and distance attenuation
                        for band, air_abs in zip(rir_bands, self.air_absorption):
                            air_decay = np.exp(-0.5 * air_abs * distance_rir)
                            band[fdl2 : N + fdl2] *= air_decay

                    # Sum up IR'S for all the bands
                    np.sum(rir_bands, axis=0, out=ir)

                else:
                    """
                    Checks if either source or the microphone directivity belongs from the class DIRPATRir
                    """

                    ir = self.dft_scale_rir_calc(
                        src.damping,
                        dist,
                        time,
                        bws,
                        N,
                        azi_m=azimuth_m,
                        col_m=colatitude_m,
                        azi_s=azimuth_s,
                        col_s=colatitude_s,
                        src_pos=s,
                        mic_pos=m,
                    )

                self.rir[-1].append(ir)

        self.simulator_state["rir_done"] = True

    def dft_scale_rir_calc(
        self,
        attenuations,
        dist,
        time,
        bws,
        N,
        azi_m,
        col_m,
        azi_s,
        col_s,
        src_pos=0,
        mic_pos=0,
    ):
        """
        Full DFT scale RIR construction.

        This function also takes into account the FIR's of the source and receiver retrieved from the SOFA file.



        Parameters
        ----------
        attenuations: arr
            Dampings for all the image sources Shape : ( No_of_octave_band x no_img_src)
        dist : arr
            distance of all the image source present in the room from this particular mic Shape : (no_img_src)
        time : arr
            Time of arrival of all the image source Shape : (no_img_src)
        bws :
            bandwidth of all the octave bands
        N :
        azi_m : arr
            Azimuth angle of arrival of this particular mic for all image sources Shape : (no_img_src)
        col_m : arr
            Colatitude angle of arrival of this particular mic  for all image sources Shape : (no_img_src)
        azi_s : arr
            Azimuth angle of departure of this particular source for all image sources Shape : (no_img_src)
        col_s : arr
            Colatitude angle of departure of this particular source for all image sources Shape : (no_img_src)
        src_pos : int
            The particular source we are calculating RIR
        mic_pos : int
            The particular mic we are calculating RIR

        Returns
        -------
            rir : :py:class:`~numpy.ndarray`
                Constructed RIR for this particlar src mic pair .

            The constructed RIR still lacks air absorption and distance absorption because in the old pyroom these calculation happens on the octave band level.


        """

        attenuations = attenuations / dist
        alp = []
        window_length = 81

        no_imag_src = attenuations.shape[1]

        fp_im = N
        fir_length_octave_band = self.octave_bands.n_fft

        from .build_rir import (
            fast_convolution_3,
            fast_convolution_4,
            fast_window_sinc_interpolator,
        )

        rec_presence = True if (len(azi_m) > 0 and len(col_m) > 0) else False
        source_presence = True if (len(azi_s) > 0 and len(col_s) > 0) else False

        final_fir_IS_len = (
            (self.mic_array.directivity[mic_pos].filter_len_ir if (rec_presence) else 1)
            + (
                self.sources[src_pos].directivity.filter_len_ir
                if (source_presence)
                else 1
            )
            + window_length
            + fir_length_octave_band
        ) - 3

        if rec_presence and source_presence:
            resp_mic = self.mic_array.directivity[mic_pos].get_response(
                azimuth=azi_m, colatitude=col_m
            )  # Return response as an array of number of (img_sources * length of filters)
            resp_src = self.sources[src_pos].directivity.get_response(
                azimuth=azi_s, colatitude=col_s
            )

            if self.mic_array.directivity[mic_pos].filter_len_ir == 1:
                resp_mic = np.array(resp_mic).reshape(-1, 1)

            else:
                assert (
                    self.fs == self.mic_array.directivity[mic_pos].fs
                ), "Mic directivity: frequency of simulation should be same as frequency of interpolation"

            if self.sources[src_pos].directivity.filter_len_ir == 1:
                resp_src = np.array(resp_src).reshape(-1, 1)
            else:
                assert (
                    self.fs == self.sources[src_pos].directivity.fs
                ), "Source directivity:  frequency of simulation should be same as frequency of interpolation"

        else:
            if source_presence:
                assert (
                    self.fs == self.sources[src_pos].directivity.fs
                ), "Directivity source frequency of simulation should be same as frequency of interpolation"

                resp_src = self.sources[src_pos].directivity.get_response(
                    azimuth=azi_s, colatitude=col_s
                )

            elif rec_presence:
                assert (
                    self.fs == self.mic_array.directivity[mic_pos].fs
                ), "Directivity mic frequency of simulation should be same as frequency of interpolation"

                resp_mic = self.mic_array.directivity[mic_pos].get_response(
                    azimuth=azi_m, colatitude=col_m
                )

        # else:
        # txt = "No"
        # final_fir_IS_len = (fir_length_octave_band + window_length) - 1

        time_arrival_is = time  # For min phase

        # Calculating fraction delay sinc filter
        sample_frac = time_arrival_is * self.fs  # Find the fractional sample number

        ir_diff = np.zeros(N + (final_fir_IS_len))  # 2050 #600

        # Create arrays for fractional delay low pass filter, sum of {damping coeffiecients * octave band filter}, source response, receiver response.

        cpy_ir_len_1 = np.zeros((no_imag_src, final_fir_IS_len), dtype=np.complex_)
        cpy_ir_len_2 = np.zeros((no_imag_src, final_fir_IS_len), dtype=np.complex_)
        cpy_ir_len_3 = np.zeros((no_imag_src, final_fir_IS_len), dtype=np.complex_)
        cpy_ir_len_4 = np.zeros((no_imag_src, final_fir_IS_len), dtype=np.complex_)
        att_in_dft_scale = np.zeros(
            (no_imag_src, fir_length_octave_band), dtype=np.complex_
        )

        # Vectorized sinc filters

        vectorized_interpolated_sinc = np.zeros(
            (no_imag_src, window_length), dtype=np.double
        )
        vectorized_time_ip = np.array(
            [int(math.floor(sample_frac[img_src])) for img_src in range(no_imag_src)]
        )
        vectorized_time_fp = [
            sample_frac[img_src] - int(math.floor(sample_frac[img_src]))
            for img_src in range(no_imag_src)
        ]
        vectorized_time_fp = np.array(vectorized_time_fp, dtype=np.double)
        vectorized_interpolated_sinc = fast_window_sinc_interpolator(
            vectorized_time_fp, window_length, vectorized_interpolated_sinc
        )

        for i in range(no_imag_src):  # Loop through Image source
            att_in_octave_band = attenuations[:, i]
            att_in_dft_scale_ = att_in_dft_scale[i, :]

            # Interpolating attenuations given in the single octave band to a DFT scale.

            att_in_dft_scale_ = self.octave_bands.octave_band_dft_interpolation(
                att_in_octave_band,
                self.air_absorption,
                dist[i],
                att_in_dft_scale_,
                bws,
                self.min_phase,
            )

            # time_ip = int(math.floor(sample_frac[i]))  # Calculating the integer sample

            # time_fp = sample_frac[i] - time_ip  # Calculating the fractional sample

            # windowed_sinc_filter = fast_window_sinc_interpolater(time_fp)

            cpy_ir_len_1[i, : att_in_dft_scale_.shape[0]] = np.fft.ifft(
                att_in_dft_scale_
            )
            cpy_ir_len_2[i, :window_length] = vectorized_interpolated_sinc[i, :]

            if source_presence and rec_presence:
                cpy_ir_len_3[i, : resp_src[i, :].shape[0]] = resp_src[i, :]

                cpy_ir_len_4[i, : resp_mic[i, :].shape[0]] = resp_mic[i, :]

                out = fast_convolution_4(
                    cpy_ir_len_1[i, :],
                    cpy_ir_len_2[i, :],
                    cpy_ir_len_3[i, :],
                    cpy_ir_len_4[i, :],
                    final_fir_IS_len,
                )

                ir_diff[
                    vectorized_time_ip[i] : (vectorized_time_ip[i] + final_fir_IS_len)
                ] += np.real(out)

            else:
                if source_presence:
                    resp = resp_src[i, :]
                elif rec_presence:
                    resp = resp_mic[i, :]

                cpy_ir_len_3[i, : resp.shape[0]] = resp

                out = fast_convolution_3(
                    cpy_ir_len_1[i, :],
                    cpy_ir_len_2[i, :],
                    cpy_ir_len_3[i, :],
                    final_fir_IS_len,
                )

                ir_diff[
                    vectorized_time_ip[i] : (vectorized_time_ip[i] + final_fir_IS_len)
                ] += np.real(out)

        return ir_diff

    def simulate(
        self,
        snr=None,
        reference_mic=0,
        callback_mix=None,
        callback_mix_kwargs={},
        return_premix=False,
        recompute_rir=False,
    ):
        r"""
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
        """

        # import convolution routine
        from scipy.signal import fftconvolve

        # Throw an error if we are missing some hardware in the room
        if len(self.sources) == 0:
            raise ValueError("There are no sound sources in the room.")
        if self.mic_array is None:
            raise ValueError("There is no microphone in the room.")

        # compute RIR if necessary
        if self.rir is None or len(self.rir) == 0 or recompute_rir:
            self.compute_rir()

        # number of mics and sources
        M = self.mic_array.M
        S = len(self.sources)

        # compute the maximum signal length
        from itertools import product

        max_len_rir = np.array(
            [len(self.rir[i][j]) for i, j in product(range(M), range(S))]
        ).max()
        f = lambda i: len(self.sources[i].signal) + np.floor(
            self.sources[i].delay * self.fs
        )
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
                premix_signals[s, m, d : d + len(sig) + len(h) - 1] += fftconvolve(
                    h, sig
                )

        if callback_mix is not None:
            # Execute user provided callback
            signals = callback_mix(premix_signals, **callback_mix_kwargs)
            self.sigma2_awgn = None

        elif snr is not None:
            # Normalize all signals so that
            denom = np.std(premix_signals[:, reference_mic, :], axis=1)
            premix_signals /= denom[:, None, None]
            signals = np.sum(premix_signals, axis=0)

            # Compute the variance of the microphone noise
            self.sigma2_awgn = 10 ** (-snr / 10) * S

        else:
            signals = np.sum(premix_signals, axis=0)

        # add white gaussian noise if necessary
        if self.sigma2_awgn is not None:
            signals += np.random.normal(0.0, np.sqrt(self.sigma2_awgn), signals.shape)

        # record the signals in the microphones
        self.mic_array.record(signals, self.fs)

        if return_premix:
            return premix_signals

    def direct_snr(self, x, source=0):
        """Computes the direct Signal-to-Noise Ratio"""

        if source >= len(self.sources):
            raise ValueError("No such source")

        if self.sources[source].signal is None:
            raise ValueError("No signal defined for source " + str(source))

        if self.sigma2_awgn is None:
            return float("inf")

        x = np.array(x)
        sigma2_s = np.mean(self.sources[0].signal ** 2)
        d2 = np.sum((x - self.sources[source].position) ** 2)

        return sigma2_s / self.sigma2_awgn / (16 * np.pi**2 * d2)

    def get_wall_by_name(self, name):
        """
        Returns the instance of the wall by giving its name.

        Parameters
        ----------
        name: string
            name of the wall

        Returns
        -------
        Wall
            instance of the wall with this name
        """

        if name in self.wallsId:
            return self.walls[self.wallsId[name]]
        else:
            raise ValueError("The wall " + name + " cannot be found.")

    def get_bbox(self):
        """Returns a bounding box for the room"""

        lower = np.amin(np.concatenate([w.corners for w in self.walls], axis=1), axis=1)
        upper = np.amax(np.concatenate([w.corners for w in self.walls], axis=1), axis=1)

        return np.c_[lower, upper]

    def is_inside(self, p, include_borders=True):
        """
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
        """

        p = np.array(p)
        if self.dim != p.shape[0]:
            raise ValueError("Dimension of room and p must match.")

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
        bbox_max_dist = np.linalg.norm(bbox[:, 1] - bbox[:, 0]) / 2

        # re-run until we get a non-ambiguous result
        it = 0
        while it < constants.get("room_isinside_max_iter"):
            # Get random point outside the bounding box
            random_vec = np.random.randn(self.dim)
            random_vec /= np.linalg.norm(random_vec)
            p0 = bbox_center + 2 * bbox_max_dist * random_vec

            ambiguous = False  # be optimistic
            is_on_border = False  # we have to know if the point is on the boundary
            count = 0  # wall intersection counter
            for i in range(len(self.walls)):
                # intersects, border_of_wall, border_of_segment = self.walls[i].intersects(p0, p)
                # ret = self.walls[i].intersects(p0, p)
                loc = np.zeros(self.dim, dtype=np.float32)
                ret = self.walls[i].intersection(p0, p, loc)

                if (
                    ret == int(Wall.Isect.ENDPT) or ret == 3
                ):  # this flag is True when p is on the wall
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

        return False

        # We should never reach this
        raise ValueError(
            """
                Error could not determine if point is in or out in maximum number of iterations.
                This is most likely a bug, please report it.
                """
        )

    def wall_area(self, wall):
        """Computes the area of a 3D planar wall.

        Parameters
        ----------
        wall: Wall instance
            the wall object that is defined in 3D space

        """

        # Algo : http://geomalgorithms.com/a01-_area.

        # Recall that the wall corners have the following shape :
        # [  [x1, x2, ...], [y1, y2, ...], [z1, z2, ...]  ]

        c = wall.corners
        n = wall.normal / np.linalg.norm(wall.normal)

        if len(c) != 3:
            raise ValueError("The function wall_area3D only supports ")

        sum_vect = [0.0, 0.0, 0.0]
        num_vertices = len(c[0])

        for i in range(num_vertices):
            sum_vect = sum_vect + np.cross(c[:, (i - 1) % num_vertices], c[:, i])

        return abs(np.dot(n, sum_vect)) / 2.0

    def get_volume(self):
        """
        Computes the volume of the room

        Returns
        -------
        float
            the volume of the room
        """
        wall_sum = 0.0

        for w in self.walls:
            n = (w.normal) / np.linalg.norm(w.normal)
            one_point = w.corners[:, 0]

            wall_sum += np.dot(n, one_point) * w.area()

        return wall_sum / 3.0

    @property
    def volume(self):
        return self.get_volume()

    @property
    def n_mics(self):
        return len(self.mic_array) if self.mic_array is not None else 0

    @property
    def n_sources(self):
        return len(self.sources) if self.sources is not None else 0

    def rt60_theory(self, formula="sabine"):
        """
        Compute the theoretical reverberation time (RT60) for the room.

        Parameters
        ----------
        formula: str
            The formula to use for the calculation, 'sabine' (default) or 'eyring'
        """

        rt60 = 0.0

        if self.is_multi_band:
            bandwidths = self.octave_bands.get_bw()
        else:
            bandwidths = [1.0]

        V = self.volume
        S = np.sum([w.area() for w in self.walls])
        c = self.c

        for i, bw in enumerate(bandwidths):
            # average absorption coefficients
            a = 0.0
            for w in self.walls:
                if len(w.absorption) == 1:
                    a += w.area() * w.absorption[0]
                else:
                    a += w.area() * w.absorption[i]
            a /= S

            try:
                m = self.air_absorption[i]
            except:
                m = 0.0

            if formula == "eyring":
                rt60_loc = rt60_eyring(S, V, a, m, c)
            elif formula == "sabine":
                rt60_loc = rt60_sabine(S, V, a, m, c)
            else:
                raise ValueError("Only Eyring and Sabine's formulas are supported")

            rt60 += rt60_loc * bw

        rt60 /= np.sum(bandwidths)
        return rt60

    def measure_rt60(self, decay_db=60, plot=False):
        """
        Measures the reverberation time (RT60) of the simulated RIR.

        Parameters
        ----------
        decay_db: float
            This is the actual decay of the RIR used for the computation. The
            default is 60, meaning that the RT60 is exactly what we measure.
            In some cases, the signal may be too short  to measure 60 dB decay.
            In this case, we can specify a lower value. For example, with 30
            dB, the RT60 is twice the time measured.
        plot: bool
            Displays a graph of the Schroeder curve and the estimated RT60.

        Returns
        -------
        ndarray (n_mics, n_sources)
            An array that contains the measured RT60 for all the RIR.
        """

        rt60 = np.zeros((self.n_mics, self.n_sources))

        for m in range(self.n_mics):
            for s in range(self.n_sources):
                rt60[m, s] = measure_rt60(
                    self.rir[m][s], fs=self.fs, plot=plot, decay_db=decay_db
                )

        return rt60
