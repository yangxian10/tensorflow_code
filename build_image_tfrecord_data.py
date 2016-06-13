#coding:utf-8
__author__ = '15072585_yx'
__date__ = '2016-6-12'
'''
build image TFRecord data
use it like:
python build_image_tfrecord_data.py \
  --train_directory=mydata/train \
  --validation_directory=mydata/validation \
  --output_directory=mydata/tf_record \
  --labels_file=mydata/labels.txt
'''

from datetime import datetime
import os
import random
import sys
import threading
import glob
import cv2

import numpy as np
import tensorflow as tf

# global variables
tf.app.flags.DEFINE_string('train_directory', '/tmp/', 'Training data directory')
tf.app.flags.DEFINE_string('validation_directory', '/tmp/', 'Validation data directory')
tf.app.flags.DEFINE_string('output_directory', '/tmp/', 'Output data directory')
tf.app.flags.DEFINE_integer('train_shards', 1024, 'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 128, 'Number of shards in validation TFRecord files.')
tf.app.flags.DEFINE_integer('num_threads', 8, 'Number of threads to preprocess the images.')
tf.app.flags.DEFINE_string('labels_file', '', 'Labels file')
FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convert_to_example(filename, label, text):
	colorspace = 'RGB'
	image_format = 'JPEG'
	'''
	image_buffer = open(filename, 'r').read()
	'''
	image = cv2.imread(filename, cv2.IMREAD_COLOR)
	height = image.shape[0]
	width = image.shape[1]
	channels = image.shape[2]
	image_buffer = image.tostring()

	example = tf.train.Example(features=tf.train.Features(feature={
		'image/height': _int64_feature(height),
		'image/width': _int64_feature(width),
		'image/colorspace': _bytes_feature(colorspace),
		'image/channels': _int64_feature(channels),
		'image/class/label': _int64_feature(label),
		'image/class/text': _bytes_feature(text),
		'image/format': _bytes_feature(image_format),
		'image/filename': _bytes_feature(os.path.basename(filename)),
		'image/encoded': _bytes_feature(image_buffer)}))
	return example

def _process_image_files_batch(thread_index, ranges, name, filenames, texts, labels, num_shards):
	num_threads = len(ranges)
	num_shards_per_batch = int(num_shards / num_threads)
	shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1], num_shards_per_batch+1).astype(int)
	num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]
	count = 0
	for i in xrange(num_shards_per_batch):
		shard = thread_index * num_shards_per_batch + i
		output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
		output_file = os.path.join(FLAGS.output_directory, output_filename)
		writer = tf.python_io.TFRecordWriter(output_file)
		files_in_shard = np.arange(shard_ranges[i], shard_ranges[i+1], dtype=int)
		for j, k in enumerate(files_in_shard):
			filename = filenames[k]
			label = labels[k]
			text = texts[k]
			example = _convert_to_example(filename, label, text)
			writer.write(example.SerializeToString())
			count += 1
			if count % 1000 == 0:
				print('%s [thread %d]: Processed %d of %d images in thread batch.' % (datetime.now(), thread_index, count, num_files_in_thread))
				sys.stdout.flush()
		print('%s [thread %d]: Wrote %d images to %s.' % (datetime.now(), thread_index, len(files_in_shard), output_file))
		sys.stdout.flush()
		writer.close()
	print('%s [thread %d]: Wrote %d images to %s shards.' % (datetime.now(), thread_index, count, num_files_in_thread))
	sys.stdout.flush()

def _process_image_files(name, filenames, texts, labels, num_shards):
	spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
	ranges = []
	threads = []
	for i in xrange(len(spacing) - 1):
		ranges.append([spacing[i], spacing[i+1]])

	print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
	sys.stdout.flush()

	coord = tf.train.Coordinator()
	threads = []
	for thread_index in xrange(len(ranges)):
		args = (thread_index, ranges, name, filenames, texts, labels, num_shards)
		t = threading.Thread(target=_process_image_files_batch, args=args)
		t.start()
		threads.append(t)
	coord.join(threads)
	print('%s: Finished writing all %d images in data set.' % (datetime.now(), len(filenames)))
	sys.stdout.flush()

def _find_image_files(data_dir, labels_file):
	print('Determining list of input files and labels from %s.' % data_dir)
	f = open(labels_file, 'rb')
	unique_labels = [line.strip() for line in f.readlines()]
	labels = []
	filenames = []
	texts = []

	for i, text in enumerate(unique_labels):
		img_path = os.path.join(data_dir, text)
		matching_files = glob.glob(os.path.join(img_path, '*.jpg'))
		labels.extend([i] * len(matching_files))
		texts.extend([text] * len(matching_files))
		filenames.extend(matching_files)
		if i % 100 == 0 and i > 0:
			print('finished finding files in %d of %d classes.' % (i, len(labels)))

	shuffled_index = range(len(filenames))
	random.seed(15072585)
	random.shuffle(shuffled_index)
	filenames = [filenames[i] for i in shuffled_index]
	texts = [texts[i] for i in shuffled_index]
	labels = [labels[i] for i in shuffled_index]

	print('found %d JPEG files across %d labels inside %s.' % (len(filenames), len(unique_labels), data_dir))
	return filenames, texts, labels	

def _process_dataset(name, directory, num_shards, labels_file):
	filenames, texts, labels = _find_image_files(directory, labels_file)
	_process_image_files(name, filenames, texts, labels, num_shards)

def main(unused_argv):
	print('Saving results to %s' % FLAGS.output_directory)
	_process_dataset('train', FLAGS.train_directory, FLAGS.train_shards, FLAGS.labels_file)
	_process_dataset('validation', FLAGS.validation_directory, FLAGS.validation_shards, FLAGS.labels_file)

if __name__ == '__main__':
	tf.app.run()