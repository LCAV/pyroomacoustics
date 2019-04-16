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

import os
import tarfile
import bz2

try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen


def download_uncompress(url, path='.', compression=None, context=None):
    '''
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
    '''

    # infer compression from url
    if compression is None:
        compression = os.path.splitext(url)[1][1:]

    # check compression format and set mode
    if compression in ['gz', 'bz2']:
        mode = 'r|' + compression
    elif compression == 'tar':
        mode = 'r:'
    else:
        raise ValueError('The file must be of type tar/gz/bz2.')

    # download and untar/uncompress at the same time
    if context is not None:
        stream = urlopen(url, context=context)
    else:
        stream = urlopen(url)
    tf = tarfile.open(fileobj=stream, mode=mode)
    tf.extractall(path)
