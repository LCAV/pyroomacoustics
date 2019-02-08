# This file contains objects and data for wall absorption parameters
# Copyright (C) 2019  Robin Scheibler
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
r"""
Material Properties
-------------------

Different materials have different absorbant and scattering coefficients.
We define a class to hold these values. The values are typically measured for octave-bands
at 125, 250, 500, 1k, 2k, 4k, and sometimes 8k.

The values given here are taken from the annex of the book

Michael Vorlaender, Auralization: Fundamentals of Acoustics, Modelling,
Simulation, Algorithms, and Acoustic Virtual Reality, Springer, 1st Edition,
2008.
"""

from collections import namedtuple

materials_table = {
        'hard_surface' : {
            'description' :  'Walls, hard surfaces average (brick walls, plaster, hard floors, etc.)',
            'absorption' : [ 0.02, 0.02, 0.03, 0.03, 0.04, 0.05, 0.05 ],
            'scattering' : None,
            'center_freqs' : [125, 250, 500, 1000, 2000, 4000, 8000],
            },
        'brickwork' : {
            'description' : 'Walls, rendered brickwork',
            'absorption' : [ 0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04 ],
            'scattering' : None,
            'center_freqs' : [125, 250, 500, 1000, 2000, 4000, 8000],
            },
        'rough_concrete' : {
            'description' : 'Rough concrete',
            'absorption' : [ 0.02, 0.03, 0.03, 0.03, 0.04, 0.07, 0.07 ],
            'scattering' : None,
            'center_freqs' : [125, 250, 500, 1000, 2000, 4000, 8000],
            },
        'smooth_concrete' : {
            'description' : 'Smooth unpainted concrete',
            'absorption' : [ 0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05 ],
            'scattering' : None,
            'center_freqs' : [125, 250, 500, 1000, 2000, 4000, 8000],
            },
        }

# Create object like structures
materials = dict([(name, namedtuple('Material', fields.keys())(**fields)) for name, fields in materials_table.items()])

# Create a type based on the above dictionary
Material = type(materials[next(iter(materials))])

class FlatAbsorber(Material):

    def __init__(self, absorption, scattering):
        self.name = 'Flat absorption over frequency'
        self.center_freqs = ['0']
        self.absorption = [absorption]
        self.scattering = [scattering]
