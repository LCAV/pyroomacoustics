'''
The LOCATA Dataset
==================

This dataset was released as a challenge in spring 2018. The goal of the
dataset is to provide a standard baseline for acoustic localization and
tracking algorithm with both dynamic and static sources.

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

def parse(line):
    m = RE_TS.match(line)
    if m:
        elem = [int(m.group(n)) for n in range(1,7)]  # year month day hour minute seconds
        elem.append(int(float('0.' + m.group(7)) * 1e6))  # microseconds
        ts = dt.datetime(*elem)

        sub = line.split()

        for n in range(6):
            sub.pop(0)

        if len(sub) == 0:
            return ts
        elif len(sub) == 1:  # that only happens for the required time file
            return ts, dict(valid=sub[0])

        ref_vec = np.array([sub.pop(0) for n in range(3)])
        rot_mat = np.array([sub.pop(0) for n in range(9)]).reshape((3,3), order='F')

        if len(sub) == 0:
            return ts, dict(ref=ref_vec, rot=rot_mat)

        array_pos = np.array(sub).reshape((3,-1), order='F')

        return ts, dict(ref=ref_vec, rot=rot_mat, pos=array_pos,)

    else:
        return None

def read_reference_file(fn):
    ''' Reads and parse a groundtruth file '''
    with open(fn, 'r') as f:
        return list(filter(lambda x:x, map(parse, f.readlines())))


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
    separate subfolder `eval` and `dev`.

    Attributes
    ----------
    basedir: str, option
        The directory where the CMU ARCTIC corpus is located/downloaded. By
        default, this is the current directory.
    recordings: list of CMUArcticSentence
        The list of all utterances in the corpus


    Parameters
    ----------
    basedir: str, optional
        The directory where the CMU ARCTIC corpus is located/downloaded. By
        default, this is the current directory.
    task: int, optional
        The number of the task to read
    rec: int or list of ints, optional
        The recordings to consider
    array: str or list of str, optional
        The arrays to read in
    dev: bool
        Set to True to restrict to dev data, False to eval data. Not setting
        the parameter will result in using both sets.

    '''
    def __init__(self, basedir=None, verbose=False, **kwargs):

        self.basedir = basedir if basedir is not None else '.'
        files = os.listdir(self.basedir)
        self.samples = []  # sample is for compatibility with the base classe
        self.recordings = self.samples  # this is for convenience
        eval_dir = os.path.join(self.basedir, 'eval')
        dev_dir = os.path.join(self.basedir, 'dev')

        if not os.path.exists(eval_dir) or not os.path.exists('dev'):
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
        Wether this a development or evaluation recording
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
                    ts = read_reference_file(os.path.join(path, FMT_TS_SOURCE.format(source=name)))
                    pos = read_reference_file(
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
        self.ts = read_reference_file(
                os.path.join(path, FMT_TS_ARRAY.format(array=array))
                )
        self.pos = read_reference_file(
                os.path.join(path, FMT_POS_ARRAY.format(array=array))
                )
        self.req = read_reference_file(
                os.path.join(path, FMT_REQ_TIME)
                )
        fs, data = wavfile.read(
                os.path.join(path, FMT_AUDIO_ARRAY.format(array=array))
                )
        AudioSample.__init__(self, data, fs, task=task, rec=rec, array=array, dev=dev)
