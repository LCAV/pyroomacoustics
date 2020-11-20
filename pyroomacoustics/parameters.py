# This file contains code related to physical properties of the room
# Copyright (C) 2015-2019  Robin Scheibler
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

"""
This file defines the main physical constants of the system:
    * Speed of sound
    * Absorption of materials
    * Scattering coefficients
    * Air absorption
"""
import io
import json
import os

import numpy as np

# tolerance for computations
eps = 1e-10


# We implement the constants as a dictionary so that they can
# be modified at runtime.
# The class Constants gives an interface to update the value of
# constants or add new ones.
_constants = {}
_constants_default = {
    "c": 343.0,  # speed of sound at 20 C in dry air
    "ffdist": 10.0,  # distance to the far field
    "fc_hp": 300.0,  # cut-off frequency of standard high-pass filter
    "frac_delay_length": 81,  # Length of the fractional delay filters used for RIR gen
    "room_isinside_max_iter": 20,  # Max iterations for checking if point is inside room
}


class Constants:
    """
    A class to provide easy access package wide to user settable constants.
    """

    def set(self, name, val):
        # add constant to dictionnary
        _constants[name] = val

    def get(self, name):

        try:
            v = _constants[name]
        except KeyError:
            try:
                v = _constants_default[name]
            except KeyError:
                raise NameError(name + ": no such constant")

        return v


# the instanciation of the class
constants = Constants()

# Compute the speed of sound as a function
# of temperature, humidity, and pressure
def calculate_speed_of_sound(t, h, p):
    """
    Compute the speed of sound as a function of
    temperature, humidity and pressure

    Parameters
    ----------
    t: float
        temperature [Celsius]
    h: float
        relative humidity [%]
    p: float
        atmospheric pressure [kpa]

    Returns
    -------

    Speed of sound in [m/s]
    """

    # using crude approximation for now
    return 331.4 + 0.6 * t + 0.0124 * h


def _calculate_temperature(c, h):
    """ Compute the temperature give a speed of sound ``c`` and humidity ``h`` """

    return (c - 331.4 - 0.0124 * h) / 0.6


r"""
Air Absorption Coefficients
---------------------------

Air absorbs sound as `exp(-distance * a)` where `distance` is the distance
travelled by sound and `a` is the absorption coefficient.
The values are measured for octave-bands at 125, 250, 500, 1k, 2k, 4k, and 8k.

The values given here are taken from the annex of the book

Michael Vorlaender, Auralization: Fundamentals of Acoustics, Modelling,
Simulation, Algorithms, and Acoustic Virtual Reality, Springer, 1st Edition,
2008.
"""

# Table of air absorption coefficients
air_absorption_table = {
    "10C_30-50%": [x * 1e-3 for x in [0.1, 0.2, 0.5, 1.1, 2.7, 9.4, 29.0]],
    "10C_50-70%": [x * 1e-3 for x in [0.1, 0.2, 0.5, 0.8, 1.8, 5.9, 21.1]],
    "10C_70-90%": [x * 1e-3 for x in [0.1, 0.2, 0.5, 0.7, 1.4, 4.4, 15.8]],
    "20C_30-50%": [x * 1e-3 for x in [0.1, 0.3, 0.6, 1.0, 1.9, 5.8, 20.3]],
    "20C_50-70%": [x * 1e-3 for x in [0.1, 0.3, 0.6, 1.0, 1.7, 4.1, 13.5]],
    "20C_70-90%": [x * 1e-3 for x in [0.1, 0.3, 0.6, 1.1, 1.7, 3.5, 10.6]],
    "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000],
}


class Physics(object):
    """
    A Physics object allows to compute the room physical properties depending
    on temperature and humidity.

    Parameters
    ----------
    temperature: float, optional
        The room temperature
    humidity: float in range (0, 100), optional
        The room relative humidity in %. Default is 0.
    """

    def __init__(self, temperature=None, humidity=None):

        self.p = 100.0  # pressure in kilo-Pascal (kPa), not used
        if humidity is None:
            self.H = 0.0
        else:
            self.H = humidity

        if self.H < 0.0 or self.H > 100:
            raise ValueError("Relative humidity is a value between 0 and 100.")

        if temperature is None:
            self.T = _calculate_temperature(constants.get("c"), self.H)
        else:
            self.T = temperature

    def get_sound_speed(self):
        """
        Returns
        -------
        the speed of sound
        """
        return calculate_speed_of_sound(self.T, self.H, self.p)

    def get_air_absorption(self):
        """
        Returns
        -------
        ``(air_absorption, center_freqs)`` where ``air_absorption`` is a list
        corresponding to the center frequencies in ``center_freqs``
        """

        key = ""

        if self.T < 15:
            key += "10C_"
        else:
            key = "20C_"

        if self.H < 50:
            key += "30-50%"
        elif 50 <= self.H and self.H < 70:
            key += "50-70%"
        else:
            key += "70-90%"

        return {
            "coeffs": air_absorption_table[key],
            "center_freqs": air_absorption_table["center_freqs"],
        }

    @classmethod
    def from_speed(cls, c):
        """ Choose a temperature and humidity matching a desired speed of sound """

        H = 0.3
        T = _calculate_temperature(c, H)

        return cls(temperature=T, humidity=H)


r"""
Material Properties
-------------------

Different materials have different absorbant and scattering coefficients.
We define a class to hold these values. The values are typically measured for
octave-bands at 125, 250, 500, 1k, 2k, 4k, and sometimes 8k.

The values given here are taken from the annex of the book

Michael Vorlaender, Auralization: Fundamentals of Acoustics, Modelling,
Simulation, Algorithms, and Acoustic Virtual Reality, Springer, 1st Edition,
2008.

"""
# the file containing the database of materials
_materials_database_fn = os.path.join(os.path.dirname(__file__), "data/materials.json")

materials_absorption_table = {
    "anechoic": {"description": "Anechoic material", "coeffs": [1.0]},
}

materials_scattering_table = {
    "no_scattering": {"description": "No scattering", "coeffs": [0.0]},
}


with io.open(_materials_database_fn, "r", encoding="utf8") as f:
    materials_data = json.load(f)

    center_freqs = materials_data["center_freqs"]

    tables = {
        "absorption": materials_absorption_table,
        "scattering": materials_scattering_table,
    }

    for key, table in tables.items():
        for subtitle, contents in materials_data[key].items():
            for keyword, p in contents.items():
                table[keyword] = {
                    "description": p["description"],
                    "coeffs": p["coeffs"],
                    "center_freqs": center_freqs[: len(p["coeffs"])],
                }


class Material(object):
    """
    A class that describes the energy absorption and scattering
    properties of walls.

    Attributes
    ----------
    energy_absorption: dict
        A dictionary containing keys ``description``, ``coeffs``, and
        ``center_freqs``.
    scattering: dict
        A dictionary containing keys ``description``, ``coeffs``, and
        ``center_freqs``.

    Parameters
    ----------
    energy_absorption: float, str, or dict
        * float: The material created will be equally absorbing at all frequencies
            (i.e. flat).
        * str: The absorption values will be obtained from the database.
        * dict: A dictionary containing keys ``description``, ``coeffs``, and
            ``center_freqs``.
    scattering: float, str, or dict
        * float: The material created will be equally scattering at all frequencies
            (i.e. flat).
        * str: The scattering values will be obtained from the database.
        * dict: A dictionary containing keys ``description``, ``coeffs``, and
            ``center_freqs``.
    """

    def __init__(self, energy_absorption, scattering=None):

        # Handle the energy absorption input based on its type
        if isinstance(energy_absorption, (float, np.float32, np.float64)):
            # This material is flat over frequencies
            energy_absorption = {"coeffs": [energy_absorption]}

        elif isinstance(energy_absorption, str):
            # Get the coefficients from the database
            energy_absorption = dict(materials_absorption_table[energy_absorption])

        elif not isinstance(energy_absorption, dict):
            raise TypeError(
                "The energy absorption of a material can be defined by a scalar value "
                "for a flat absorber, a name refering to a material in the database, "
                "or a list with one absoption coefficients per frequency band"
            )

        if scattering is None:
            # By default there is no scattering
            scattering = 0.0

        if isinstance(scattering, (float, np.float32, np.float64)):
            # This material is flat over frequencies
            # We match the number of coefficients for the absorption
            if len(energy_absorption["coeffs"]) > 1:
                scattering = {
                    "coeffs": [scattering] * len(energy_absorption["coeffs"]),
                    "center_freqs": energy_absorption["center_freqs"],
                }
            else:
                scattering = {"coeffs": [scattering]}

        elif isinstance(scattering, str):
            # Get the coefficients from the database
            scattering = dict(materials_scattering_table[scattering])

        elif not isinstance(scattering, dict):
            # In all other cases, the material should be a dictionary
            raise TypeError(
                "The scattering of a material can be defined by a scalar value "
                "for a flat absorber, a name refering to a material in the database, "
                "or a list with one absoption coefficients per frequency band"
            )

        # Now handle the case where energy absorption is flat, but scattering is not
        if len(scattering["coeffs"]) > 1 and len(energy_absorption["coeffs"]) == 1:
            n_coeffs = len(scattering["coeffs"])
            energy_absorption["coeffs"] = energy_absorption["coeffs"] * n_coeffs
            energy_absorption["center_freqs"] = list(scattering["center_freqs"])

        # checks for `energy_absorption` dict
        assert isinstance(energy_absorption, dict), (
            "`energy_absorption` must be a "
            "dictionary with the keys "
            "`coeffs` and `center_freqs`."
        )
        assert "coeffs" in energy_absorption.keys(), (
            "Missing `coeffs` keys in " "`energy_absorption` dict."
        )
        if len(energy_absorption["coeffs"]) > 1:
            assert len(energy_absorption["coeffs"]) == len(
                energy_absorption["center_freqs"]
            ), (
                "Length of `energy_absorption['coeffs']` and "
                "energy_absorption['center_freqs'] must match."
            )

        # checks for `scattering` dict
        assert isinstance(scattering, dict), (
            "`scattering` must be a "
            "dictionary with the keys "
            "`coeffs` and `center_freqs`."
        )
        assert "coeffs" in scattering.keys(), (
            "Missing `coeffs` keys in " "`scattering` dict."
        )
        if len(scattering["coeffs"]) > 1:
            assert len(scattering["coeffs"]) == len(scattering["center_freqs"]), (
                "Length of `scattering['coeffs']` and "
                "scattering['center_freqs'] must match."
            )

        self.energy_absorption = energy_absorption
        self.scattering = scattering

    def is_freq_flat(self):
        """
        Returns ``True`` if the material has flat characteristics over
        frequency, ``False`` otherwise.
        """
        return (
            len(self.energy_absorption["coeffs"]) == 1
            and len(self.scattering["coeffs"]) == 1
        )

    @property
    def absorption_coeffs(self):
        """ shorthand to the energy absorption coefficients """
        return self.energy_absorption["coeffs"]

    @property
    def scattering_coeffs(self):
        """ shorthand to the scattering coefficients """
        return self.scattering["coeffs"]

    def resample(self, octave_bands):
        """ resample at given octave bands """
        self.energy_absorption = {
            "coeffs": octave_bands(**self.energy_absorption),
            "center_freqs": octave_bands.centers,
        }
        self.scattering = {
            "coeffs": octave_bands(**self.scattering),
            "center_freqs": octave_bands.centers,
        }

    @classmethod
    def all_flat(cls, materials):
        """
        Checks if all materials in a list are frequency flat

        Parameters
        ----------
        materials: list or dict of Material objects
            The list of materials to check

        Returns
        -------
        ``True`` if all materials have a single parameter, else ``False``
        """
        if isinstance(materials, dict):
            return all([m.is_freq_flat() for m in materials.values()])
        else:
            return all([m.is_freq_flat() for m in materials])


def make_materials(*args, **kwargs):
    """
    Helper method to conveniently create multiple materials.

    Each positional and keyword argument should be a valid input
    for the Material class. Then, for each of the argument, a
    Material will be created by calling the constructor.

    If at least one positional argument is provided, a list of
    Material objects constructed using the provided positional
    arguments is returned.

    If at least one keyword argument is provided, a dict with keys
    corresponding to the keywords and containing Material objects
    constructed with the keyword values is returned.

    If only positional arguments are provided, only the list is returned.
    If only keyword arguments are provided, only the dict is returned.
    If both are provided, both are returned.
    If no argument is provided, an empty list is returned.
    """

    ret_args = []
    for parameters in args:
        if isinstance(parameters, (list, tuple)):
            ret_args.append(Material(*parameters))
        else:
            ret_args.append(Material(parameters))

    ret_kwargs = {}
    for name, parameters in kwargs.items():
        if isinstance(parameters, (list, tuple)):
            ret_kwargs[name] = Material(*parameters)
        else:
            ret_kwargs[name] = Material(parameters)

    if len(ret_kwargs) == 0:
        return ret_args
    elif len(ret_args) == 0:
        return ret_kwargs
    else:
        return ret_args, ret_kwargs
