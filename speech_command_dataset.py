'''
The Tenserflow Speech Command Dataset
=====================================

'''

import os
import numpy as np 
from scipy.io import wavfile

try:
	import sounddevice as sd 
	have_sounddevice = True
except:
	have_sounddevice = False

from .utils import download_uncompress_tar_bz2
from .base import Meta, AudioSample, Dataset

#The sound code and attributes
tenserflow_sounds = {
	'01' : { 'sound' : 'zero', 'lang' : 'US English'},
	'02' : { 'sound' : 'yes', 'lang' : 'US English'},
	'03' : { 'sound' : 'wow', 'lang' : 'US English'},
	'04' : { 'sound' : 'up', 'lang' : 'US English'},
	'05' : { 'sound' : 'two', 'lang' : 'US English'},
	'06' : { 'sound' : 'tree', 'lang' : 'US English'},
	'07' : { 'sound' : 'stop', 'lang' : 'US English'},
	'08' : { 'sound' : 'six', 'lang' : 'US English'},
	'09' : { 'sound' : 'sheila', 'lang' : 'US English'},
	'10' : { 'sound' : 'seven', 'lang' : 'US English'},
	'11' : { 'sound' : 'right', 'lang' : 'US English'},
	'12' : { 'sound' : 'one', 'lang' : 'US English'},
	'13' : { 'sound' : 'on', 'lang' : 'US English'},
	'14' : { 'sound' : 'off', 'lang' : 'US English'},
	'15' : { 'sound' : 'no', 'lang' : 'US English'},
	'16' : { 'sound' : 'nine', 'lang' : 'US English'},
	'17' : { 'sound' : 'marvin', 'lang' : 'US English'},
	'18' : { 'sound' : 'left', 'lang' : 'US English'},
	'19' : { 'sound' : 'house', 'lang' : 'US English'},
	'20' : { 'sound' : 'happy', 'lang' : 'US English'},
	'21' : { 'sound' : 'go', 'lang' : 'US English'},
	'22' : { 'sound' : 'four', 'lang' : 'US English'},
	'23' : { 'sound' : 'five', 'lang' : 'US English'},
	'24' : { 'sound' : 'eight', 'lang' : 'US English'},
	'25' : { 'sound' : 'down', 'lang' : 'US English'},
	'26' : { 'sound' : 'dog', 'lang' : 'US english'},
	'27' : { 'sound' : 'cat', 'lang' : 'US english'},
	'28' : { 'sound' : 'bird', 'lang' : 'US english'},
	'29' : { 'sound' : 'bed', 'lang' : 'US English'},
	'30' : { 'sound' : '_background_noise_', 'lang' : 'Noise'},
	}

#The sounds in the database
tenserflow_sounds_data = {}

#directory structure
sounds_dir = 'tenserflow_{}_dataset'

#download info
url_base = 'http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz'.format(sounds_dir)

class TenserflowCorpus(Dataset):
	'''
    This class will load the CMU ARCTIC corpus in a
    structure amenable to be processed.

    Attributes
    -----------
    basedir: str, option
        The directory where the Tenserflow corpus is located/downloaded. By
        default, this is the current directory.
    info: dict
        A dictionary whose keys are the labels of metadata fields attached to the samples.
        The values are lists of all distinct values the field takes.
    sound_list : list of TenserflowSound
        The list of all utterances in the corpus

    Parameters
    ----------
    basedir: str, option
        The directory where the Tenserflow corpus is located/downloaded. By
        default, this is the current directory.
    download: bool, optional
        if the corpus does not exist, download it.
    spoken: str or list of str , optional
        A list of the Tenserflow different sound labels. If provided, only those sounds are loaded.
        By default, all sounds are loaded.
    lang: str or list of str, optional
        The kanguage, only 'English' is avaible here4
     '''
     def __init__(self,basedir=None, download=false, build=True, **kwargs):

     	#initialize
     	Dataset.__init__(self)

        # we give a meaningful alias to the sample list from the base class
     	self.sound_list = self.samples

     	self.basedir = basedir
     	if basedir is None:
     		self.basedir = './Tenserflow_dataset'

     	if 'spoken'  not in kwargs:
     		kwargs['spoken'] = list(tenserflow_sounds.keys())

     	spoken_sounds = kwargs['spoken']
     	if not isinstance(spoken_sounds, list):
     		spoken_sounds = [spoken_sounds]

     	if not os.path.exists(self.basedir):
     		if download:
     			os.mkdir(self.basedir)
     		else:
     			raise ValueError('Corpus directory does not exist. Create or set download option.')

     	n_spoken_sounds = len(spoken_sounds)
     	spoken_sounds = [spoken_sound for spoken_sound in spoken_sounds if spoken_sound in tenserflow_sounds.keys()]
        if n_spoken_sounds != len(spoken_sounds):
        	import warnings
        	warnings.warn('Some invalid speakers were removed from the list.', RuntimeWarning)

        for spoken_sound in spoken_sounds:
        	sdir = os.path.join(self.basedir,sounds_dir.format(spoken_sound))

        	if not os.path.exists(sdir):
        		if download:
        			url = url_base.format(spoken_sound)
        			print('Download', url, 'into', self.basedir, '...')
        			download_uncompress_tar_bz2(url,self.basedir)
        		else:
        			raise ValueError('Missing spoken_sound directory. Please Download.')

        for spoken_sound in spoken_sounds:
        	sdir = os.path.join(self.basedir, sounds_dir.format(spoken_sound))

        	all_files = []

        	with open(os.path.join(sdir, 'testing_list.txt'),'r') as f:
        		for line in f.readlines():
        			l = line.split('/')
        			tag = l[1]
        			sound = l[0]
        			path = os.path.join(sdir,sound + '/' + tag)

        			if tag not in tenserflow_sounds_data:
        				tenserflow_sounds_data[tag] = {
        				         'sound': sound,
        				         'paths': {spoken_sound : path}
        				}
        			else:
        				tenserflow_sounds_data[tag]['paths'][spoken_sound] = path


        if build:
        	self.build_corpus(**kwargs)

    def build_corpus(self,**kwargs):

    	'''
    	Build the corpus with some filters
    	'''

    	for tag, info in tenserflow_sounds_data.items():
    		for spoken_sound, path in info['paths'].items():
    			meta = Meta(spoken_sound=spoken_sound, tag=tag, sound=info['sound'], **tenserflow_sounds_data[spoken_sound])

    			if meta.match(**kwargs):
    				self.add_sample(TenserflowSound(path, **meta.as_dict()))

    def filter(self, **kwargs):
    	'''
    	Filter the corpus and selects samples that match the criteria provided
    	'''

    	new_corpus = TenserflowCorpus(basedir=self.basedir, build=False, spoken_sound=[])

    	for s in self.samples:
    		new_corpus.add_sample_matching(s, **kwargs)

    	return new_corpus



class TenserflowSound(AudioSample):

	'''
	Create the sound object

	Parameters
	----------
	path: str
	  the path to the audio file
	**kwargs:
	  metadata as a list of keyword arguments

	Attributes
	----------
	data: array_like
	  the actual audio signal
	fs: int
	  sampling frequency
	'''

	def __init__(self,path,**kwargs):
		'''
		Create the the sound object
		path: string
		  the path to a particular sample
		'''

		fs,data = wavfile.read(path)

		AudioSample.__init__(self, data, fs, **kwargs)

	def __str__(self):
		''' String representation'''
		template = '{spoken_sound} ({sound},{lang}); {tag}: ''{sound}'''
		s = template.format(**self.meta.as_dict())
		return s

    def plot(self,**kwargs):
    	'''Plot the spectogram'''
    	try:
    		import matplotlib.pyplot as plt 
    	except ImportError:
    		print('Warning: matplotlib is required for plotting')
    		return
    	AudioSample.plot(self,**kwargs)
    	plt.title(self.meta.sound)

		








		
