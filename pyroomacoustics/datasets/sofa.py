# Some classes to apply rotate objects or indicate directions in 3D space.
# Copyright (C) 2024 Robin Scheibler
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
Pyroomacoustics contains a small database of SOFA files that have been tested
and can be used for simultions.
The database can be loaded using the function
:py:class:`~pyroomacoustics.datasets.sofa.SOFADatabase`.

.. code-block:: python

    # load and display the list of available SOFA files and their content
    from pyroomacoustics.datasets import SOFADatabase

    db = SOFADatabase()
    db.list()

The database contains the following files.

- Three files from the `DIRPAT database
  <https://phaidra.kug.ac.at/detail/o:68229#?page=1&pagesize=10&collection=o:67857>`_
  collected by Manuel Brandner, Matthias Frank, and Daniel Rudrich University
  of Music and Performing Arts, Graz.

  - ``AKG_c480_c414_CUBE.sofa`` containing the directive responses of a
    microphone with 4 different patterns.
  - ``EM32_Directivity.sofa`` that contains the directional response of the
    `Eigenmike em32 <https://eigenmike.com/eigenmike-em32>`_ microphone array.
  - ``LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa`` that contains 12 source
    directivities. This file is dynamically downloaded upon its first use.
  - The files are public domain
    (`CC0 <https://creativecommons.org/public-domain/cc0/>`_),
    but if you use them in your research, please cite the following
    `paper <https://aes2.org/publications/elibrary-page/?id=19538>`_.

    ::

        M. Brandner, M. Frank, and D. Rudrich, "DirPat—Database and
        Viewer of 2D/3D Directivity Patterns of Sound Sources and Receivers,"
        in Audio Engineering Society Convention 144, Paper 425, 2018.

- Two head-related transfer functions of the MIT KEMAR dummy head
  with normal and large pinna. The data was collected by Bill Gardner
  and Keith Martin from MIT and is free to use provided the authors are
  cited. See the
  `full license <https://sound.media.mit.edu/resources/KEMAR/README>`_
  for more details.


"""


import json
import typing as tp
from dataclasses import dataclass
from pathlib import Path

from .utils import AttrDict, download_multiple

_pra_data_folder = Path(__file__).parents[1] / "data"
DEFAULT_SOFA_PATH = _pra_data_folder / "sofa"
SOFA_INFO = _pra_data_folder / "sofa_files.json"

_DIRPAT_FILES = [
    "Soundfield_ST450_CUBE",
    "AKG_c480_c414_CUBE",
    "Oktava_MK4012_CUBE",
    "LSPs_HATS_GuitarCabinets_Akustikmessplatz",
]


def is_dirpat(name):
    if isinstance(name, Path):
        name = name.stem
    return name in _DIRPAT_FILES


def get_sofa_db():
    """
    A helper function to quickly load the SOFA database
    """
    # we want to avoid loading the database multiple times
    global sofa_db
    try:
        return sofa_db
    except NameError:
        sofa_db = SOFADatabase()
        return sofa_db


def resolve_sofa_path(path):
    path = Path(path)

    if path.exists():
        return path

    sofa_db = get_sofa_db()
    if path.stem in sofa_db:
        return Path(sofa_db[path.stem].path)

    raise ValueError(f"SOFA file {path} could not be found")


def get_sofa_db_info():
    with open(SOFA_INFO, "r") as f:
        sofa_info = json.load(f)
    return sofa_info


def download_sofa_files(path=None, overwrite=False, verbose=False, no_fail=False):
    """
    Download the SOFA files containing source/receiver impulse responses

    Parameters
    ----------
    path: str or Path, optional
        A path to a directory where the files will be downloaded
    overwrite: bool, optional
        If set to `True`, forces the download even if the files already exist
    verbose: bool
        Print some information about the download status (default `False`)

    Returns
    -------
    files: list of Path
        The list of path to the files downloaded
    """
    if path is None:
        path = DEFAULT_SOFA_PATH

    path = Path(path)

    path.mkdir(exist_ok=True, parents=True)

    sofa_info = get_sofa_db_info()

    files = {
        path / f"{name}.sofa": info["url"]
        for name, info in sofa_info.items()
        if info["supported"]
    }
    download_multiple(files, overwrite=overwrite, verbose=verbose, no_fail=no_fail)

    return list(files.keys())


@dataclass
class SOFAFileInfo:
    """
    A class to store information about a SOFA file

    Parameters
    ----------
    path: Path
        The path to the SOFA file
    supported: bool
        Whether the SOFA file is supported by Pyroom Acoustics
    type: str
        The type of device (e.g., 'sources' or 'microphones')
    url: str
        The URL where the SOFA file can be downloaded
    homepage: str
        The URL of the SOFA file homepage
    license: str
        The license of the SOFA file
    contains: List[str]
        The labels of the sources/microphones contained in the SOFA file,
        or``None`` if the information is not available
    """

    path: Path
    supported: bool = True
    type: str = "unknown"
    url: str = "unknown"
    homepage: str = "unknown"
    license: str = "unknown"
    contains: tp.List[str] = None


class SOFADatabase(dict):
    """
    A small database of SOFA files containing source/microphone directional
    impulse responses

    The database object is a dictionary-like object where the keys are the
    names of the SOFA files and the values are objects with the following
    attributes:

    .. code-block:: python

        db = SOFADatabase()

        # type of device: 'sources' or 'microphones'
        db["Soundfield_ST450_CUBE"].type

        # list of the labels of the sources/microphones
        db["Soundfield_ST450_CUBE"].contains


    Parameters
    ----------
    download: bool, optional
        If set to `True`, the SOFA files are downloaded if they are not already
        present in the default folder
    """

    def __init__(self, download=True):
        super().__init__()

        if download:
            # specify "no_fail" to avoid errors if internet is not available
            download_sofa_files(path=self.root, no_fail=True)

        self._db = {}
        for name, info in get_sofa_db_info().items():
            path = self.root / f"{name}.sofa"
            if path.exists():
                dict.__setitem__(self, name, SOFAFileInfo(path=path, **info))

        for path in DEFAULT_SOFA_PATH.glob("*.sofa"):
            name = path.stem
            if name not in self:
                dict.__setitem__(
                    self,
                    name,
                    SOFAFileInfo(path=path),
                )

    def list(self):
        """
        Print a list of the available SOFA files and the labels of the
        different devices they contain
        """
        for name, info in self.items():
            print(f"- {name} ({info.type})")
            if info.contains is not None:
                for channel in info.contains:
                    print(f"  - {channel}")

    @property
    def root(self):
        """The path to the folder containing the SOFA files"""
        return DEFAULT_SOFA_PATH

    @property
    def db_info_path(self):
        """The path to the JSON file containing the SOFA files information"""
        return SOFA_INFO

    def __setitem__(self, key, val):
        # disallow writing elements
        raise RuntimeError(f"{self.__class__} is not writable")
