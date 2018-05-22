
import os, tarfile, bz2, requests

try:
    from urllib.request import urlopen
except ImportError:
    from urllib import urlopen

def download_uncompress(url, path='.', compression=None):
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
    stream = urlopen(url)
    tf = tarfile.open(fileobj=stream, mode=mode)
    tf.extractall(path)
