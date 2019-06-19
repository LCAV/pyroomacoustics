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
}


class Constants:
    """
    A class to provide easy access package wide to user settable constants.

    Be careful of not using this in tight loops since it uses exceptions.
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
        The room relative humidity in %
    """

    def __init__(self, temperature=None, humidity=0.):

        self.p = 100.0  # pressure in kilo-Pascal (kPa), not used
        self.H = humidity

        if self.H < 0. or self.H > 100:
            raise ValueError("Relative humidity is a value between 0 and 100.")

        if temperature is None:
            temperature = _calculate_temperature(constants.get("c"), self.H)
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

materials_absorption_table = {
    "anechoic": {"description": "Anechoic material", "coeffs": [1.0]},
    "hard_surface": {
        "description": "Walls, hard surfaces average (brick walls, plaster, "
                       "hard floors, etc.)",
        "coeffs": [0.02, 0.02, 0.03, 0.03, 0.04, 0.05, 0.05],
        "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000],
    },
    "brickwork": {
        "description": "Walls, rendered brickwork",
        "coeffs": [0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04],
        "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000],
    },
    "rough_concrete": {
        "description": "Rough concrete",
        "coeffs": [0.02, 0.03, 0.03, 0.03, 0.04, 0.07, 0.07],
        "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000],
    },
    "smooth_concrete": {
        "description": "Smooth unpainted concrete",
        "coeffs": [0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05],
        "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000],
    },
    "6mm_carpet": {
        "description": "(Floor covering) 6 mm pile carpet bonded to "
                       "closed-cell foam underlay",
        "coeffs": [0.03, 0.09, 0.25, 0.31, 0.33, 0.44, 0.44],
        "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000],
    },
}

materials_scattering_table = {
    "no_scattering": {"description": "No scattering", "coeffs": [0.0]},
    "rpg_skyline": {
        "description": "Diffuser RPG Skyline",
        "coeffs": [0.01, 0.08, 0.45, 0.82, 1.0],
        "center_freqs": [125, 250, 500, 1000, 2000],
    },
    "rpg_qrd": {
        "description": "Diffuser RPG QRD",
        "coeffs": [0.06, 0.15, 0.45, 0.95, 0.88, 0.91],
        "center_freqs": [125, 250, 500, 1000, 2000, 4000],
    },
    "theatre_audience": {
        "description": "Theatre Audience",
        "coeffs": [0.3, 0.5, 0.6, 0.6, 0.7, 0.7, 0.7],
        "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000],
    },
    "classroom_tables": {
        "description": "Rows of classroom tables and persons on chairs",
        "coeffs": [0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.6],
        "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000],
    },
    "amphitheatre_steps": {
        "description": "Amphitheatre steps, length 82 cm, height 30 cm "
                       "(Farnetani 2005)",
        "coeffs": [0.05, 0.45, 0.75, 0.9, 0.9],
        "center_freqs": [125, 250, 500, 1000, 2000],
    },
}


class Material(object):
    """
    A class to access materials

    Attributes
    ----------
    absorption: dict
        A dictionary containing keys ``description``, ``coeffs``, and
        ``center_freqs``.
    scattering: dict
        A dictionary containing keys ``description``, ``coeffs``, and
        ``center_freqs``.
    """

    def __init__(self, absorption, scattering):

        # checks for `absorption` dict
        assert isinstance(absorption, dict), '`absorption` must be a ' \
                                             'dictionary with the keys ' \
                                             '`coeffs` and `center_freqs`.'
        assert 'coeffs' in absorption.keys(), 'Missing `coeffs` keys in ' \
                                              '`absorption` dict.'
        if len(absorption['coeffs']) > 1:
            assert len(absorption['coeffs']) == \
                   len(absorption['center_freqs']), \
                "Length of `absorption['coeffs']` and " \
                "absorption['center_freqs'] must match."

        # checks for `scattering` dict
        assert isinstance(scattering, dict), '`scattering` must be a ' \
                                             'dictionary with the keys ' \
                                             '`coeffs` and `center_freqs`.'
        assert 'coeffs' in scattering.keys(), 'Missing `coeffs` keys in ' \
                                              '`scattering` dict.'
        if len(scattering['coeffs']) > 1:
            assert len(scattering['coeffs']) == \
                   len(scattering['center_freqs']), \
                "Length of `scattering['coeffs']` and " \
                "scattering['center_freqs'] must match."

        self.absorption = absorption
        self.scattering = scattering

    def is_freq_flat(self):
        """
        Returns ``True`` if the material has flat characteristics over
        frequency, ``False`` otherwise.
        """
        return (
            len(self.absorption["coeffs"]) == 1 and
            len(self.scattering["coeffs"]) == 1
        )

    def get_abs(self):
        """ shorthand to the absorption coefficients """
        return self.absorption["coeffs"]

    def get_scat(self):
        """ shorthand to the scattering coefficients """
        return self.scattering["coeffs"]

    def resample(self, octave_bands):
        """ resample at given octave bands """
        self.absorption = {
            "coeffs": octave_bands(**self.absorption),
            "center_freqs": octave_bands.centers,
        }
        self.scattering = {
            "coeffs": octave_bands(**self.scattering),
            "center_freqs": octave_bands.centers,
        }

    @classmethod
    def from_db(cls, abs_name, scat_name="no_scattering"):
        """
        Constructs a ``Material`` object from names of entries in the materials
        database.

        Parameters
        ----------
        abs_name: str
            Name of absorbing material
        scat_name: str, optional
            Name of scattering characteristic (default: ``no_scattering``)
        """
        return cls(
            materials_absorption_table[abs_name],
            materials_scattering_table[scat_name]
        )

    @classmethod
    def make_freq_flat(cls, absorption=0.0, scattering=0.0):
        """
        Construct a material with flat characteristics over frequency

        Parameters
        ----------
        absorption: float
            The absorption coefficient
        scattering: float
            The scattering coefficient
        """
        return cls({"coeffs": [absorption]}, {"coeffs": [scattering]})

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
