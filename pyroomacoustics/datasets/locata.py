'''
The LOCATA Dataset
==================

This dataset was released as a challenge in spring 2018. The goal of the
dataset is to provide a standard baseline for acoustic localization and
tracking algorithm with both dynamic and static sources. In particular,
there are six tasks.

* **Task 1**: Localization of a single, static loudspeaker using static microphone arrays
* **Task 2**: Localization of multiple static loudspeakers using static microphone arrays
* **Task 3**: Tracking of a single, moving talker using static microphone arrays
* **Task 4**: Tracking of multiple, moving talkers using static microphone arrays
* **Task 5**: Tracking of a single, moving talker using moving microphone arrays
* **Task 6**: Tracking of multiple moving talkers using moving microphone arrays.

The data is released under a very permissive license, however, at the time of
writing of this code, the `official download <http://www.locata-challenge.org>`_
page requires registration to obtain the dataset. No open mirror currently exists.

License: `Open Data Commons Attribution License <https://opendatacommons.org/licenses/by/1-0/>`_

Price: Free

URL: http://www.locata-challenge.org
'''
import os, re
import numpy as np
from scipy.io import wavfile
import datetime as dt
import warnings

from ..doa import cart2spher

from .base import Meta, AudioSample, Dataset

url = 'http://www.locata-challenge.org'

# the regexp for the dataset metadata
RE_PATH = re.compile('(eval|dev)/task([1-6])/recording([0-9]+)/(benchmark2|dicit|dummy|eigenmike)$')
RE_AUDIO_SOURCE = re.compile('audio_source_([A-Za-z0-9]+).wav')
FMT_POS_SOURCE = 'position_source_{source}.txt'
FMT_TS_SOURCE = 'audio_source_timestamps_{source}.txt'
FMT_POS_ARRAY = 'position_array_{array}.txt'
FMT_AUDIO_ARRAY = 'audio_array_{array}.wav'
FMT_TS_ARRAY = 'audio_array_timestamps_{array}.txt'
FMT_REQ_TIME = 'required_time.txt'
RE_TS = re.compile('^(20[0-9]{2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{2})\.(\d+)')

def _parse(line):
    #m = RE_TS.match(line)
    sub = line.split()
    if len(sub) >= 6 and sub[0] == '2017':
        elem = [int(sub.pop(0)) for n in range(5)]
        frac_sec = float(sub.pop(0))
        elem.append(int(frac_sec))  # seconds
        elem.append(int((frac_sec - elem[-1]) * 1e6))  # microseconds
        #elem = [int(m.group(n)) for n in range(1,7)]  # year month day hour minute seconds
        #elem.append(int(float('0.' + m.group(7)) * 1e6))  # microseconds
        ts = dt.datetime(*elem)

        #sub = line.split()

        #for n in range(6):
        #    sub.pop(0)

        if len(sub) == 0:
            return dict(ts=ts)
        elif len(sub) == 1:  # that only happens for the required time file
            return dict(ts=ts, valid=int(sub[0]))

        point = np.array([float(sub.pop(0)) for n in range(3)])
        ref_vec = np.array([float(sub.pop(0)) for n in range(3)])
        rot_mat = np.array([float(sub.pop(0)) for n in range(9)]).reshape((3,3))

        if len(sub) == 0:
            return dict(ts=ts, point=point, ref_vec=ref_vec, rot_mat=rot_mat)

        mics_pos = np.array([float(s) for s in sub]).reshape((3,-1), order='F')

        return dict(ts=ts, point=point, ref_vec=ref_vec, rot_mat=rot_mat, mics=mics_pos,)

    else:
        return None

def _read_reference_file(fn):
    ''' Reads and parse a groundtruth file '''
    with open(fn, 'r') as f:
        return list(filter(lambda x:x, map(_parse, f.readlines())))


def _find_ts(L, T):
    ''' find the closest timestamp in a list '''
    return np.argmin(np.abs(list(map(lambda x: (x['ts'] - T).total_seconds(), L))))



locata_tasks = [1, 2, 3, 4, 5 , 6]
locata_arrays = [
        'benchmark2',
        'dicit',
        'dummy',
        'eigenmike',
        ]

class LOCATA(Dataset):
    '''
    This class wraps the LOCATA challenge dataset.

    The development and evaluation datasets are released in folders with similar
    names for task1 to task6.  We assume that the user will have saved these in
    separate subfolder ``eval`` and ``dev``.

    Attributes
    ----------
    basedir: str, option
        The directory where the LOCATA dataset is located/downloaded. By
        default, this is the current directory.
    recordings: list of LocataRecording
        The list of all utterances in the corpus


    Parameters
    ----------
    basedir: str, optional
        The directory where the LOCATA dataset is located/downloaded. By
        default, this is the current directory.
    task: int or list of ints, optional
        The number of the task to read
    rec: int or list of ints, optional
        The recordings to consider
    array: str or list of str, optional
        The arrays to read in
    dev: bool
        Set to ``True`` to restrict to dev data, ``False`` to eval data. Not setting
        the parameter will result in using both sets.

    '''
    def __init__(self, basedir=None, verbose=False, **kwargs):

        self.basedir = basedir if basedir is not None else '.'
        files = os.listdir(self.basedir)
        self.samples = []  # sample is for compatibility with the base classe
        self.recordings = self.samples  # this is for convenience
        eval_dir = os.path.join(self.basedir, 'eval')
        dev_dir = os.path.join(self.basedir, 'dev')

        if not os.path.exists(eval_dir) or not os.path.exists(dev_dir):
            warnings.warn('The ''eval'' and/or ''dev'' folders are missing. Please check the structure of the dataset directory.')


        for path, dirs, files in os.walk(self.basedir):

            m = RE_PATH.search(path)

            if m:
                dev = False if m.group(1) == 'eval' else True
                task, rec, array = int(m.group(2)), int(m.group(3)), m.group(4)
                meta = Meta(task=task, rec=rec, array=array, dev=dev)
                if array in locata_arrays and meta.match(**kwargs):
                    if verbose:
                        print(path)
                    self.samples.append(
                            LocataRecording(path, task=task, rec=rec, array=array, dev=dev)
                            )

        if len(self) == 0:
            warnings.warn('Nothing was imported. The dataset can be downloaded at ' + url + '.')



class LocataRecording(AudioSample):
    '''
    An instance of a recording in the LOCATA challenge dataset.
    This class inherits all the methods from :py:obj:`pyroomacoustics.datasets.base.AudioSample`.

    Attributes
    ----------
    data: array_like
        The actual data
    fs: int
        The sampling frequency of the input signal
    meta: pyroomacoustics.datasets.Meta
        An object containing the sample metadata. They can be accessed using the
        dot operator
    sources: dict or None
        A dictionary containing the reference audio recording, as well as the
        timestamps and location information for the sources (only available for
        the `dev` dataset)
    ts: list of datetime.DateTime
        A list of timestamps corresponding to the audio samples
    pos: list
        A list of timestamps and position information for the microphone array
        at each time instant
    req: list
        The list of timestamps at which the source location needs to be computed

    Parameters
    ----------
    path: str
        The path to the recording folder
    task: int
        The task number
    rec: int
        The recording number
    array: str
        The array used
    dev: bool
        Whether this a development or evaluation recording
    '''

    def __init__(self, path, task=None, rec=None, array=None, dev=True):

        self.path = path

        if dev:
            self.sources = dict()

            # Read the sources
            for fn in os.listdir(path):
                m = RE_AUDIO_SOURCE.search(fn)
                if m:
                    name = m.group(1)
                    fs, data = wavfile.read(os.path.join(path, fn))
                    ts = _read_reference_file(os.path.join(path, FMT_TS_SOURCE.format(source=name)))
                    pos = _read_reference_file(
                            os.path.join(path, FMT_POS_SOURCE.format(source=name))
                            )
                    self.sources[name] = dict(
                            audio=AudioSample(data, fs, name=name),
                            ts=ts,
                            pos=pos,
                            )
        else:
            self.sources = None

        # read the array info
        self.ts = _read_reference_file(
                os.path.join(path, FMT_TS_ARRAY.format(array=array))
                )
        self.pos = _read_reference_file(
                os.path.join(path, FMT_POS_ARRAY.format(array=array))
                )
        self.req = _read_reference_file(
                os.path.join(path, FMT_REQ_TIME)
                )
        fs, data = wavfile.read(
                os.path.join(path, FMT_AUDIO_ARRAY.format(array=array))
                )
        AudioSample.__init__(self, data, fs, task=task, rec=rec, array=array, dev=dev)


    def get_sampleindex(self, ts):
        '''
        Find the sample index closest to a given timestamp.

        Parameters
        ----------
        ts: datetime
            The timestamp
        '''
        return _find_ts(self.ts, ts)


    def get_ts(self, ts):
        '''
        Get the audio sample timestamp closest to a given
        datetime object or a sample index.

        Parameters
        ----------
        ts: int or datetime
            The timestamp or audio sample index

        Returns
        -------
        A datetime object or None if the sample doesn't exist
        '''

        if type(ts) == int:

            if ts < 0 or ts > len(self.ts):
                return None
            else:
                ts = self.ts[ts]['ts']

        return ts


    def get_array(self, ts):
        '''
        Get the location of the microphones in the local reference frame

        Parameters
        ----------
        ts: int or datetime
            The timestamp or audio sample index

        Returns
        -------
        An ndarray containing the microphone locations in the columns.
        '''

        ts = self.get_ts(ts)

        arr_i = _find_ts(self.pos, ts)
        array_pos = self.pos[arr_i]['point']
        array_rot = self.pos[arr_i]['rot_mat']
        array_mics = self.pos[arr_i]['mics']

        return np.dot(array_rot.T, array_mics - array_pos[:,None])


    def get_doa(self, ts):
        '''
        Returns the doa of sources at a given time. This is only available for
        ``dev`` recordings.

        Parameters
        ----------
        ts: int or datetime
            The timestamp or audio sample index

        Returns
        -------
        A dictionary containing the spherical coordinates (radius, azimuth,
        colatitude) of the sources with respect to the array or None if the
        timestamp is unavailable, or the sources information is unavailable
        (eval recordings).
        '''

        if not self.meta.dev:
            return None

        ts = self.get_ts(ts)

        arr_i = _find_ts(self.pos, ts)
        array_pos = self.pos[arr_i]['point']
        array_rot = self.pos[arr_i]['rot_mat']

        doa = dict()

        for name, info in self.sources.items():

            src_i = _find_ts(info['pos'], ts)
            src_pos = info['pos'][src_i]['point']
            v = np.dot(array_rot.T, src_pos - array_pos)
            doa[name] = dict(zip(['radius', 'azimuth', 'colatitude'], cart2spher(v)))

            # fix the azimuth
            doa[name]['azimuth'] -= np.pi / 2
            if doa[name]['azimuth'] > np.pi:
                doa[name]['azimuth'] -= 2 * np.pi

        return doa
