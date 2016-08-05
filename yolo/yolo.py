#coding:utf-8
__author__ = '15072585_yx'
__date__ = '2016-7-8'
'''
tensorflow yolo
'''

import numpy as np
import tensorflow as tf
import skimage.io
import skimage.transform 

import time
import cv2
import sys, getopt
import glob
import os

class YOLO_TF:
	def __init__(self, weights_file):
		self.classes = ["men_jacket", "men_bottem", "underwear", "women_bottem", "women_jacket", "men_shoes", "women_shoes", "shuma", "drink", "food", "shipin", "xihua"]
		self.threshold = 0.2
		self.iou_threshold = 0.5
		self.num_class = len(self.classes)
		self.num_box = 2
		self.grid_size = 7
		self.negative_slope = 0.1

		self.disp_console = True
		self.imshow = True

		self.img_width = 448
		self.img_height = 448

		self.x = tf.placeholder(tf.float32, [None, self.img_width, self.img_height, 3])

		'''
		self.W = {
			'wc1': self.weight_variable([7,7,3,64]),
			'wc2': self.weight_variable([3,3,64,192]),
			'wc3': self.weight_variable([1,1,192,128]),
			'wc4': self.weight_variable([3,3,128,256]),
			'wc5': self.weight_variable([1,1,256,256]),
			'wc6': self.weight_variable([3,3,256,512]),
			'wc7': self.weight_variable([1,1,512,256]),
			'wc8': self.weight_variable([3,3,256,512]),
			'wc9': self.weight_variable([1,1,512,256]),
			'wc10': self.weight_variable([3,3,256,512]),
			'wc11': self.weight_variable([1,1,512,256]),
			'wc12': self.weight_variable([3,3,256,512]),
			'wc13': self.weight_variable([1,1,512,256]),
			'wc14': self.weight_variable([3,3,256,512]),
			'wc15': self.weight_variable([1,1,512,512]),
			'wc16': self.weight_variable([3,3,512,1024]),
			'wc17': self.weight_variable([1,1,1024,512]),
			'wc18': self.weight_variable([3,3,512,1024]),
			'wc19': self.weight_variable([1,1,1024,512]),
			'wc20': self.weight_variable([3,3,512,1024]),
			'wc21': self.weight_variable([3,3,1024,1024]),
			'wc22': self.weight_variable([3,3,1024,1024]),
			'wc23': self.weight_variable([3,3,1024,1024]),
			'wc24': self.weight_variable([3,3,1024,1024]),
			'wfc25': self.weight_variable([7*7*1024,4096]),
			'wfc26': self.weight_variable([4096,self.grid_size*self.grid_size*(5*self.num_box+self.num_class)])
			}
		self.B = {
			'bc1': self.bias_variable([64]),
			'bc2': self.bias_variable([192]),
			'bc3': self.bias_variable([128]),
			'bc4': self.bias_variable([256]),
			'bc5': self.bias_variable([256]),
			'bc6': self.bias_variable([512]),
			'bc7': self.bias_variable([256]),
			'bc8': self.bias_variable([512]),
			'bc9': self.bias_variable([256]),
			'bc10': self.bias_variable([512]),
			'bc11': self.bias_variable([256]),
			'bc12': self.bias_variable([512]),
			'bc13': self.bias_variable([256]),
			'bc14': self.bias_variable([512]),
			'bc15': self.bias_variable([512]),
			'bc16': self.bias_variable([1024]),
			'bc17': self.bias_variable([512]),
			'bc18': self.bias_variable([1024]),
			'bc19': self.bias_variable([512]),
			'bc20': self.bias_variable([1024]),
			'bc21': self.bias_variable([1024]),
			'bc22': self.bias_variable([1024]),
			'bc23': self.bias_variable([1024]),
			'bc24': self.bias_variable([1024]),
			'bfc25': self.bias_variable([4096]),
			'bfc26': self.bias_variable([self.grid_size*self.grid_size*(5*self.num_box+self.num_class)])
			}
		'''
		self.params = np.load(weights_file).item()
		self.pred = self.create_yolo(self.x)
		self.sess = tf.Session()
		self.sess.run(tf.initialize_all_variables())

	def __del__(self):
		self.sess.close()

	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
		return tf.Variable(initial)

	def bias_variable(self, shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial, trainable=True)

	def conv2d(self, x, W, B, stride, name):
		with tf.name_scope(name) as scope:
			# tips: padding can't be 'SAME', must use tf.pad and 'VALID' !!!
			pad_size = W.shape[0]//2
			pad_mat = np.array([[0,0], [pad_size,pad_size],[pad_size,pad_size], [0,0]])
			x_pad = tf.pad(x, pad_mat)
			conv = tf.nn.conv2d(x_pad, W, strides=[1, stride, stride, 1], padding='VALID')
			bias = tf.nn.bias_add(conv, B)
			#conv = tf.nn.relu(bias*self.negative_slope, name=scope)
			conv = tf.maximum(bias, bias*self.negative_slope)
			return conv

	def max_pool(self, x, k, name):
		return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

	def avg_pool(self, x, k, name):
		return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

	def create_yolo(self, x):
		conv1 = self.conv2d(x, self.params['conv1']['weights'], self.params['conv1']['biases'], 2, 'conv1')
		pool1 = self.max_pool(conv1, 2, 'pool1')
		#print pool1.get_shape()

		conv2 = self.conv2d(pool1, self.params['conv2']['weights'], self.params['conv2']['biases'], 1, 'conv2')
		pool2 = self.max_pool(conv2, 2, 'pool2')
		#print pool2.get_shape()

		conv3 = self.conv2d(pool2, self.params['conv3']['weights'], self.params['conv3']['biases'], 1, 'conv3')
		conv4 = self.conv2d(conv3, self.params['conv4']['weights'], self.params['conv4']['biases'], 1, 'conv4')
		conv5 = self.conv2d(conv4, self.params['conv5']['weights'], self.params['conv5']['biases'], 1, 'conv5')
		conv6 = self.conv2d(conv5, self.params['conv6']['weights'], self.params['conv6']['biases'], 1, 'conv6')
		pool6 = self.max_pool(conv6, 2, 'pool6')
		#print pool6.get_shape()

		conv7 = self.conv2d(pool6, self.params['conv7']['weights'], self.params['conv7']['biases'], 1, 'conv7')
		conv8 = self.conv2d(conv7, self.params['conv8']['weights'], self.params['conv8']['biases'], 1, 'conv8')
		conv9 = self.conv2d(conv8, self.params['conv9']['weights'], self.params['conv9']['biases'], 1, 'conv9')
		conv10 = self.conv2d(conv9, self.params['conv10']['weights'], self.params['conv10']['biases'], 1, 'conv10')
		conv11 = self.conv2d(conv10, self.params['conv11']['weights'], self.params['conv11']['biases'], 1, 'conv11')
		conv12 = self.conv2d(conv11, self.params['conv12']['weights'], self.params['conv12']['biases'], 1, 'conv12')
		conv13 = self.conv2d(conv12, self.params['conv13']['weights'], self.params['conv13']['biases'], 1, 'conv13')
		conv14 = self.conv2d(conv13, self.params['conv14']['weights'], self.params['conv14']['biases'], 1, 'conv14')
		conv15 = self.conv2d(conv14, self.params['conv15']['weights'], self.params['conv15']['biases'], 1, 'conv15')
		conv16 = self.conv2d(conv15, self.params['conv16']['weights'], self.params['conv16']['biases'], 1, 'conv16')
		pool16 = self.max_pool(conv16, 2, 'pool16')
		#print pool16.get_shape()

		conv17 = self.conv2d(pool16, self.params['conv17']['weights'], self.params['conv17']['biases'], 1, 'conv17')
		conv18 = self.conv2d(conv17, self.params['conv18']['weights'], self.params['conv18']['biases'], 1, 'conv18')
		conv19 = self.conv2d(conv18, self.params['conv19']['weights'], self.params['conv19']['biases'], 1, 'conv19')
		conv20 = self.conv2d(conv19, self.params['conv20']['weights'], self.params['conv20']['biases'], 1, 'conv20')
		conv21 = self.conv2d(conv20, self.params['conv21']['weights'], self.params['conv21']['biases'], 1, 'conv21')
		conv22 = self.conv2d(conv21, self.params['conv22']['weights'], self.params['conv22']['biases'], 2, 'conv22')
		conv23 = self.conv2d(conv22, self.params['conv23']['weights'], self.params['conv23']['biases'], 1, 'conv23')
		conv24 = self.conv2d(conv23, self.params['conv24']['weights'], self.params['conv24']['biases'], 1, 'conv24')

		conv24_flat = tf.reshape(conv24, [-1, self.params['fc25']['weights'].shape[0]])
		fc25 = tf.matmul(conv24_flat, self.params['fc25']['weights']) + self.params['fc25']['biases']
		fc25 = tf.maximum(fc25*self.negative_slope, fc25)
		fc26 = tf.matmul(fc25, self.params['fc26']['weights']) + self.params['fc26']['biases']

		return fc26

	def interpret_output(self, output):
		w_img = self.img_width
		h_img = self.img_height
		gs = self.grid_size
		nb = self.num_box
		nc = self.num_class
		
		probs = np.zeros((gs, gs, nb, nc))
		class_probs = np.reshape(output[0:gs*gs*nc], (gs,gs,nc))
		#print class_probs
		scales = np.reshape(output[gs*gs*nc : gs*gs*nc+gs*gs*nb], (gs,gs,nb))
		#print scales
		boxes = np.reshape(output[gs*gs*nc+gs*gs*nb:], (gs,gs,nb,4))
		offset = np.transpose(np.reshape(np.array([np.arange(gs)]*14), (nb,gs,gs)), (1,nb,0))

		boxes[:,:,:,0] += offset
		boxes[:,:,:,1] += np.transpose(offset, (1,0,2))
		boxes[:,:,:,0:2] = boxes[:,:,:,0:2] / 7.0
		boxes[:,:,:,2] = np.multiply(boxes[:,:,:,2], boxes[:,:,:,2])
		boxes[:,:,:,3] = np.multiply(boxes[:,:,:,3], boxes[:,:,:,3])
			
		boxes[:,:,:,0] *= w_img
		boxes[:,:,:,1] *= h_img
		boxes[:,:,:,2] *= w_img
		boxes[:,:,:,3] *= h_img

		for i in range(nb):
			for j in range(nc):
				probs[:,:,i,j] = np.multiply(class_probs[:,:,j], scales[:,:,i])
		filter_mat_probs = np.array(probs>=self.threshold, dtype='bool')
		filter_mat_boxes = np.nonzero(filter_mat_probs)
		boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
		probs_filtered = probs[filter_mat_probs]
		classes_num_filtered = np.argmax(filter_mat_probs,axis=3)[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]] 

		argsort = np.array(np.argsort(probs_filtered))[::-1]
		boxes_filtered = boxes_filtered[argsort]
		probs_filtered = probs_filtered[argsort]
		classes_num_filtered = classes_num_filtered[argsort]

		for i in range(len(boxes_filtered)):
			if probs_filtered[i] == 0:
				continue
			for j in range(i+1, len(boxes_filtered)):
				if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold : 
					probs_filtered[j] = 0.0
			
		filter_iou = np.array(probs_filtered>0.0, dtype='bool')
		boxes_filtered = boxes_filtered[filter_iou]
		probs_filtered = probs_filtered[filter_iou]
		classes_num_filtered = classes_num_filtered[filter_iou]

		result = []
		for i in range(len(boxes_filtered)):
			result.append([self.classes[classes_num_filtered[i]], boxes_filtered[i][0], boxes_filtered[i][1], boxes_filtered[i][2], boxes_filtered[i][3], probs_filtered[i]])

		return result

	def iou(self, box1, box2):
		tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
		lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
		if tb < 0 or lr < 0 : intersection = 0
		else : intersection =  tb*lr
		return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)

	def show_results(self, file_name, results):
		img_cp = cv2.imread(file_name)
		for i in range(len(results)):
			x = int(results[i][1])
			y = int(results[i][2])
			w = int(results[i][3])//2
			h = int(results[i][4])//2
			if self.disp_console:
				print '    class : ' + results[i][0] + ' , [x,y,w,h]=[' + str(x) + ',' + str(y) + ',' + str(int(results[i][3])) + ',' + str(int(results[i][4]))+'], Confidence = ' + str(results[i][5])
			xmin = x-w
			xmax = x+w
			ymin = y-h
			ymax = y+h
			if xmin < 0:
				xmin = 0
			if ymin < 0:
				ymin = 0
			if xmax > self.img_width:
				xmax = self.img_width
			if ymax > self.img_height:
				ymax = self.img_height
			if self.imshow:
				cv2.rectangle(img_cp, (xmin,ymin), (xmax,ymax), (0,255,0), 2)
				print xmin, ymin, xmax, ymax
				cv2.rectangle(img_cp, (xmin,ymin-20), (xmax,ymin), (125,125,125), -1)
				cv2.putText(img_cp, results[i][0] + ' : %.2f' % results[i][5], (xmin+5,ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)	
		if self.imshow :
			cv2.imshow('YOLO detection', img_cp)
			cv2.waitKey(1000)

	def imread(self, file_name):
		img = skimage.img_as_float(skimage.io.imread(file_name, as_grey=False)).astype(np.float32)
		#img = img*2.0 - 1.0
		img_resize = skimage.transform.resize(img, (self.img_width, self.img_height, 3))
		img_resize = img_resize[:, :, (2,1,0)]
		#img = cv2.imread(file_name)
		#img = img/255.0
		#img_resize = cv2.resize(img, (self.img_width, self.img_height))
		return img_resize

	def detect(self, file_name):
		start = time.time()
		img = self.imread(file_name)
		img_test = img.reshape([1, img.shape[0], img.shape[1], img.shape[2]])

		net_output = self.sess.run(self.pred, feed_dict={self.x : img_test})
		#print net_output[0]
		result = self.interpret_output(net_output[0])

		end = time.time()
		if self.disp_console:
			print 'Time of processing this image: %.2f' % (end-start)
		self.show_results(file_name, result)


def main(argv):
	model_filename = ''
	weight_filename = ''
	img_filename = ''
	img_dir = ''
	try:
		opts, args = getopt.getopt(argv, "hm:w:i:d:")
		print opts
	except getopt.GetoptError:
		print 'yolo.py -w <output_file> -i <img_file>'
		print 'OR'
		print 'yolo.py -w <output_file> -d <img_dir>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'yolo.py -w <weight_file> -i <img_file>'
			print 'OR'
			print 'yolo.py -w <output_file> -d <img_dir>'
			sys.exit()
		elif opt == "-w":
			weight_filename = arg
			print 'weight file is "', weight_filename
		elif opt == "-i":
			img_filename = arg
			print 'image file is "', img_filename
		elif opt == "-d":
			img_dir = arg
			print 'image dir is "', img_dir

	yolo = YOLO_TF(weight_filename)

	if img_filename is not '':
		yolo.detect(img_filename)
		cv2.waitKey(10000)
	else:
		for img_file in glob.iglob(os.path.join(img_dir, '*.jpg')):
			yolo.detect(img_file)
			cv2.waitKey(10000)

if __name__ == '__main__':
	main(sys.argv[1:])