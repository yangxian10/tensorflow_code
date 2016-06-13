#coding:utf-8
__author__ = '15072585_yx'
__date__ = '2016-5-30'
'''
CNN demo
'''

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 64
display_step = 20

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.8 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

def weight_variable(shape):
	initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, trainable=True)

def conv2d(x, W, B, name):
	with tf.name_scope(name) as scope:
		conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
		bias = tf.nn.bias_add(conv, B)
		conv = tf.nn.relu(bias, name=scope)
		return conv

def max_pool(x, k, name):
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def avg_pool(x, k, name):
	return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def norm(x, lsize, name):
	return tf.nn.lrn(x, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

def my_net(_x, _weights, _biases, _dropout):
	_x = tf.reshape(_x, shape=[-1, 28, 28, 1])

	conv1 = conv2d(_x, _weights['wc1'], _biases['bc1'], 'conv1')
	pool1 = max_pool(conv1, k=2, name='pool1')
	norm1 = norm(pool1, lsize=4, name='norm1')

	conv2 = conv2d(norm1, _weights['wc2'], _biases['bc2'], 'conv2')
	pool2 = max_pool(conv2, k=2, name='pool2')
	norm2 = norm(pool2, lsize=4, name='norm2')

	pool2_flat = tf.reshape(norm2, [-1, _weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.nn.relu(tf.matmul(pool2_flat, _weights['wd1']) + _biases['bd1'])
	fc1_drop = tf.nn.dropout(fc1, _dropout)

	fc2 = tf.nn.relu(tf.matmul(fc1_drop, _weights['wd2']) + _biases['bd2'])
	fc2_drop = tf.nn.dropout(fc2, _dropout)

	out = tf.matmul(fc2_drop, _weights['out']) + _biases['out']
	return out

weights = {
	'wc1': weight_variable([3, 3, 1, 64]),
	'wc2': weight_variable([3, 3, 64, 128]),
	'wd1': weight_variable([7*7*128, 1024]),
	'wd2': weight_variable([1024, 1024]),
	'out': weight_variable([1024, 10])
}
biases = {
	'bc1': bias_variable([64]),
	'bc2': bias_variable([128]),
	'bd1': bias_variable([1024]),
	'bd2': bias_variable([1024]),
	'out': bias_variable([n_classes])
}

# Construct model
pred = my_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
	sess.run(init)
	step = 1# Keep training until reach max iterations
	while step * batch_size < training_iters:
		batch = mnist.train.next_batch(batch_size)
		# Fit training using batch data
		sess.run(optimizer, feed_dict={x: batch[0], y: batch[1], keep_prob: dropout})
		if step % display_step == 0:
			# Calculate batch accuracy
			acc = sess.run(accuracy, feed_dict={x: batch[0], y: batch[1], keep_prob: 1.})
			# Calculate batch loss
			loss = sess.run(cost, feed_dict={x: batch[0], y: batch[1], keep_prob: 1.})
			print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
		step += 1
	print "Optimization Finished!"
	# Calculate accuracy for 256 mnist test images
	print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.})
