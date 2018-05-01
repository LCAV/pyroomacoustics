#The Tenserflow Authors
#2017
#label_wav.py (Version 2.0) [Source Code].
#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/label_wav.py
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from scipy.io import wavfile

import argparse
import sys

import tensorflow as tf

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

FLAGS = None

def modify_input_wav(wav,room_dim,max_order,audio_dest):

	fs, audio_anechoic = wavfile.read(wav)

	#create a room
	room = pra.ShoeBox(
		room_dim,
		absorption=0.2,
		fs=fs,
		max_order = max_order
		)

	#source and mic location
	room.add_source([2,3.1,2],signal=audio_anechoic)
	room.add_microphone_array(
		pra.MicrophoneArray(
	        np.array([[2, 1.5, 2]]).T, 
	        room.fs)
	    )

	#source ism
	room.simulate()
	audio_reverb = room.mic_array.to_wav(audio_dest,norm=True ,bitdepth=np.int16)

def  load_graph(f):
	with tf.gfile.FastGFile(f,'rb') as graph:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(name_of_graph.read())
		tf.import_graph_def(graph_def, name='')

def load_labels(f):
	return [line.rstrip() for line in tf.gfile.GFile(f)]

def run_graph(wav_data, labels, input_name, output_name, how_many_labels):
	with tf.session() as session:
		softmax_tensor = session.graph.get_tensor_by_name(output_name)
		predictions, = session.run(softmax_tensor,{input_name: wav_data})

	top_k = predictions.argsort()[-how_many_labels:][::-1]
	for node_id in top_k:
		human_string = labels[node_id]
		score = predictions[node_id]
		print('%s (score = %.5f)' % (human_string, score))

	return 0

def label_wav(wav,labels,graph,input_name,output_name, how_many_labels):
	if not wav or not tf.gfile.Exists(wav):
		tf.logging.fatal('Audio file does not exist %s',wav)
	if not labels or not tf.gfile.Exists(labels):
		tf.logging.fatal('Labems file does not exist %s', labels)
	if not graph or not tf.gfile.Exists(graph):
		tf.logging.fatal('Graph file does not exist %s', graph)

	labels_list = load_labels(labels)
	load_graph(graph)

	with open(wav,'rb') as wav_file:
		wav_data = wav_file.read()

	run_graph(wav_data,labels_list,input_name,output_name,how_many_labels)

def main(_):
	print(FLAGS.room_dim)
	modify_input_wav(FLAGS.wav,FLAGS.room_dim,FLAGS.max_order,FLAGS.dest_wav)
	# label_wav(FLAGS.dest_wav, FLAGS.labels, FLAGS.graph, FLAGS.input_name, FLAGS.output_name, FLAGS.how_many_labels)




if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--wav', type=str, default='', help='the audio file you want processed and then identified.')
	parser.add_argument(
		'--graph', type=str, default='', help='the model you want to use for identification.')
	parser.add_argument(
		'--labels', type=str, default='', help='the path to the fil containing the labels for your data.')
	parser.add_argument(
		'--dest_wav', type=str, default='', help='the place where you want the processed data to be saved before using it with the model.')
	# parser.add_argument(
	# 	'--dim', '--room_dim', action='append',default=[5,4,6], help='give the different coordinates foe a 3D shoebox room.')
	parser.add_argument('--room_dim', nargs='+', type=int, default=[5,4,6], help='give the different coordinates for a 3D shoebox room.')
	parser.add_argument(
		'--max_order', type=int, default=3, help='the number of reflection you want to do.')
	parser.add_argument(
		'--input_name', type=str, default='wav_data_node', help='the name of the WAV data input node in the model.')
	parser.add_argument(
		'--output_name', type=str, default='labels_node', help='the name of the node outputting a prediction in the model.')
	parser.add_argument(
		'--how_many_labels', type=int, default=3, help='Number of result to show.')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)