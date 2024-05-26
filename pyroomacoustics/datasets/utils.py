# Utility for downloading and uncompressing dataset
# Copyright (C) 2019 Robin Scheibler, MIT License.
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

import bz2
import os
import tarfile
from pathlib import Path

try:
    from urllib.request import urlopen, urlretrieve
except ImportError:
    # support for python 2.7, should be able to remove by now
    from urllib import urlopen, urlretrieve


class AttrDict(object):
    """Convert a dictionary into an object"""

    def __init__(self, dictionary):
        for key, val in dictionary.items():
            if isinstance(val, dict):
                setattr(self, key, AttrDict(val))
            elif isinstance(val, list):
                setattr(
                    self, key, [AttrDict(v) if isinstance(v, dict) else v for v in val]
                )
            else:
                setattr(self, key, val)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, val):
        return setattr(self, key, val)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return repr(self.__dict__)


def download_uncompress(url, path=".", compression=None, context=None):
    """
    This functions download and uncompress on the fly a file
    of type tar, tar.gz, tar.bz2.

    Parameters
    ----------
    url: str
        The URL of the file
    path: str, optional
        The path where to uncompress the file
    compression: str, optional
        The compression type (one of 'bz2', 'gz', 'tar'), infered from url
        if not provided
    context: SSL certification, optional
        Default is to use none.
    """

    # infer compression from url
    if compression is None:
        compression = os.path.splitext(url)[1][1:]

    # check compression format and set mode
    if compression in ["gz", "bz2"]:
        mode = "r|" + compression
    elif compression == "tar":
        mode = "r:"
    else:
        raise ValueError("The file must be of type tar/gz/bz2.")

    # download and untar/uncompress at the same time
    if context is not None:
        stream = urlopen(url, context=context)
    else:
        stream = urlopen(url)
    tf = tarfile.open(fileobj=stream, mode=mode)
    tf.extractall(path)


def download_multiple(files_dict, overwrite=False, verbose=False, no_fail=False):
    """
    A utility to download multiple files

    Parameters
    ----------
    files_dict: dict
        A dictionary of files to download with key=local_path and value=url
    overwrite: bool
        If `True` if the local file exists, it will be overwritten. If `False`
        (default), existing files are skipped.
    """
    skip_ok = not overwrite

    for path, url in files_dict.items():
        path = Path(path)
        if path.exists() and skip_ok:
            if verbose:
                print(
                    f"{path} exists: skip. Use `overwrite` option to download anyway."
                )
            continue

        if verbose:
            print(f"Download {url} -> {path}...", end="")

        try:
            urlretrieve(url, path)
        except URLError:
            if no_fail:
                continue
            else:
                raise URLError(f"Failed to download {url}")

        if verbose:
            print(" done.")
