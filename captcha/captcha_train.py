import tensorflow as tf
from captcha_data import OCR_data
# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 64
display_step = 20

# Network Parameters
# n_input = 7200  # 30*80*3
n_classes = 62  # 10+26+26

data_train = OCR_data(1000, '/data/captcha_data')
data_test = OCR_data(500, '/data/captcha_data')

# tf Graph input
x = tf.placeholder(tf.float32, [None, 30, 80, 3])
y = tf.placeholder(tf.float32, [None, 4*n_classes])

def print_activations(t):
	print(t.op.name, t.get_shape().as_list())

def weight_variable(shape):
	initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.0, shape=shape)
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

weights = {
	'wc1': weight_variable([5, 5, 3, 32]),
	'wc2': weight_variable([5, 5, 32, 32]),
	'wc3': weight_variable([3, 3, 32, 32]),
	'wd1': weight_variable([4*10*32, 512]),
	'out1': weight_variable([512, n_classes]),
	'out2': weight_variable([512, n_classes]),
	'out3': weight_variable([512, n_classes]),
	'out4': weight_variable([512, n_classes])
}
biases = {
	'bc1': bias_variable([32]),
	'bc2': bias_variable([32]),
	'bc3': bias_variable([32]),
	'bd1': bias_variable([512]),
	'out1': bias_variable([n_classes]),
	'out2': bias_variable([n_classes]),
	'out3': bias_variable([n_classes]),
	'out4': bias_variable([n_classes]),
}

def ocr_net(_x, _weights, _biases):
	_x = tf.reshape(_x, shape=[-1, 30, 80, 3])

	conv1 = conv2d(_x, _weights['wc1'], _biases['bc1'], 'conv1')
	print_activations(conv1)
	pool1 = max_pool(conv1, k=2, name='pool1')
	print_activations(pool1)

	conv2 = conv2d(pool1, _weights['wc2'], _biases['bc2'], 'conv2')
	print_activations(conv2)
	pool2 = avg_pool(conv2, k=2, name='pool2')
	print_activations(pool2)

	conv3 = conv2d(pool2, _weights['wc3'], _biases['bc3'], 'conv3')
	print_activations(conv3)
	pool3 = avg_pool(conv3, k=2, name='pool3')
	print_activations(pool3)

	pool3_flat = tf.reshape(pool3, [-1, _weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.nn.relu(tf.matmul(pool3_flat, _weights['wd1']) + _biases['bd1'], name='fc1')
	print_activations(fc1)

	fc21 = tf.nn.relu(tf.matmul(fc1, _weights['out1']) + _biases['out1'], name='fc21')
	print_activations(fc21)

	fc22 = tf.nn.relu(tf.matmul(fc1, _weights['out2']) + _biases['out2'], name='fc22')
	print_activations(fc22)

	fc23 = tf.nn.relu(tf.matmul(fc1, _weights['out3']) + _biases['out3'], name='fc23')
	print_activations(fc23)

	fc24 = tf.nn.relu(tf.matmul(fc1, _weights['out4']) + _biases['out4'], name='fc24')
	print_activations(fc24)

	out = tf.concat(axis=1, values=[fc21, fc22, fc23, fc24], name='out')
	print_activations(out)
	return out

def accuracy_func(_pred, _y):
	y = tf.reshape(_y, shape=[-1, 4, 62])
	pred = tf.reshape(_pred, shape=[-1, 4, 62])
	correct_pred = tf.equal(tf.argmax(pred,2), tf.argmax(y,2))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	return accuracy

pred = ocr_net(x, weights, biases)

cost = tf.reduce_mean(y*tf.log(pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#correct_pred = tf.equal(tf.argmax(pred,2), tf.argmax(y,2))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
accuracy = accuracy_func(pred, y)

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
	sess.run(init)
	step = 1# Keep training until reach max iterations
	while step * batch_size < training_iters:
		batch = data_train.next_batch(batch_size)
		# Fit training using batch data
		sess.run(optimizer, feed_dict={x: batch[0], y: batch[1]})
		if step % display_step == 0:
			# Calculate batch accuracy
			acc = sess.run(accuracy, feed_dict={x: batch[0], y: batch[1]})
			# Calculate batch loss
			loss = sess.run(cost, feed_dict={x: batch[0], y: batch[1]})
			print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
		step += 1
	print "Optimization Finished!"

	test_batch = data_test.next_batch(500)
	print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_batch[0], y: test_batch[1]})