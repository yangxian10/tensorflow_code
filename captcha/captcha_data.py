import numpy as np
import cv2
import random
import tensorflow as tf
from captcha.image import ImageCaptcha

class OCR_data(object):
	def __init__(self, num, data_dir, batch_size=50, len_code=4, height=30, width=80):
		self.num = num
		self.data_dir = data_dir
		self.batch_size = batch_size
		self.len_code = len_code
		self.height = height
		self.width = width
		self.captcha = ImageCaptcha()
		self.index_in_epoch = 0
		self._imgs = []
		self._labels = []
		for i in range(self.num):
			if i % 100 == 0:
				print '%s images have been created.' % i
			img, label = self.create_captcha()
			self._imgs.append(img)
			self._labels.append(label)
		self._imgs = np.array(self._imgs)
		self._labels = np.array(self._labels)


	def create_captcha(self):
		code, label = self.gen_rand()
		img = self.captcha.generate(code)
		img = np.fromstring(img.getvalue(), dtype='uint8')
		img = cv2.imdecode(img, cv2.IMREAD_COLOR)
		img = cv2.resize(img, (self.width, self.height))
		return (img, label)

	def gen_rand(self):
		buf = ''
		label = []
		for i in range(self.len_code):
			rnd = random.randint(0, 61)
			label.append(rnd)
			if rnd < 10:
				ascii_code = chr(rnd+48)
			elif rnd < 36:
				ascii_code = chr(rnd+65)
			else:
				ascii_code = chr(rnd+97)
			buf += ascii_code
		label_one_hot = self.dense_to_one_hot(label, 62)
		return buf, label_one_hot

	def dense_to_one_hot(self, labels_dense, num_classes):
		num_labels = len(labels_dense)
		index_offest = np.arange(num_labels) * num_classes
		labels_one_hot = np.zeros((num_labels, num_classes))
		labels_one_hot.flat[index_offest + labels_dense] = 1
		labels_one_hot = labels_one_hot.reshape(num_labels*num_classes)
		return labels_one_hot

	def next_batch(self, batch_size):
		start = self.index_in_epoch
		self.index_in_epoch += batch_size
		if self.index_in_epoch > self.num:
			perm = np.arange(self.num)
			np.random.shuffle(perm)
			self._imgs = self._imgs[perm]
			self._labels = self._labels[perm]
			start = 0
			self.index_in_epoch = batch_size
			assert batch_size <= self.num
		end = self.index_in_epoch
		return self._imgs[start:end], self._labels[start:end]