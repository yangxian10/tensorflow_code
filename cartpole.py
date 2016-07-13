#coding:utf-8
__author__ = '15072585_yx'
__date__ = '2016-7-8'
'''
DQN for OpenAI CartPole
'''

import gym
import tensorflow as tf 
import numpy as np 
import random
from collections import deque

# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 10000 # Episode limitation
STEP = 300 # Step limitation in an episode
TEST = 10

# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch

class DQN:
	def __init__(self, env):
		self.replay_buffer = deque()
		self.time_step = 0
		self.epsilon = INITIAL_EPSILON
		self.state_dim = env.observation_space.shape[0]
		self.action_dim = env.action_space.n

		self.state_input = tf.placeholder('float', [None, self.state_dim])
		self.action_input = tf.placeholder('float', [None, self.action_dim])
		self.y_input = tf.placeholder('float', [None])

		self.weights = {
			'w1': self.weight_variable([self.state_dim, 20]),
			'w2': self.weight_variable([20, self.action_dim])
			}
		self.biases = {
			'b1': self.bias_variable([20]),
			'b2': self.bias_variable([self.action_dim]),
			}

		self.Q_value = self.create_Q_network(self.state_input)
		self.optimizer = self.create_training_method(self.action_input, self.y_input)

		self.session = tf.InteractiveSession()
		self.session.run(tf.initialize_all_variables())

	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
		return tf.Variable(initial)

	def bias_variable(self, shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial, trainable=True)

	def perceive(self, state, action, reward, next_state, done): # 感知信息
		one_hot_action = np.zeros(self.action_dim)
		one_hot_action[action] = 1
		self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
		if len(self.replay_buffer) > REPLAY_SIZE:
			self.replay_buffer.popleft()
		if len(self.replay_buffer) > BATCH_SIZE:
			self.train_Q_network()

	def egreedy_action(self, state): # 获取随机动作
		Q_value = self.Q_value.eval(feed_dict = {self.state_input:[state]})[0]
		if random.random() <= self.epsilon:
			action = random.randint(0, self.action_dim-1)
		else:
			action = np.argmax(Q_value)
		self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
		return action

	def action(self, state):
		return np.argmax(self.Q_value.eval(feed_dict = {self.state_input:[state]})[0])

	def create_Q_network(self, state_input):
		h_layer = tf.nn.relu(tf.matmul(state_input, self.weights['w1']) + self.biases['b1'])
		Q_value = tf.matmul(h_layer, self.weights['w2']) + self.biases['b2']
		return Q_value

	def create_training_method(self, action_input, y_input):
		Q_action = tf.reduce_sum(tf.mul(self.Q_value, action_input), reduction_indices=1)
		cost = tf.reduce_mean(tf.square(y_input - Q_action))
		optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)
		return optimizer

	def train_Q_network(self):
		self.time_step += 1

		minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
		state_batch = [x[0] for x in minibatch]
		action_batch = [x[1] for x in minibatch]
		reward_batch = [x[2] for x in minibatch]
		next_state_batch = [x[3] for x in minibatch]

		y_batch = []
		Q_value_batch = self.Q_value.eval(feed_dict={self.state_input : next_state_batch})
		for i in range(BATCH_SIZE):
			done = minibatch[i][4]
			if done:
				y_batch.append(reward_batch[i])
			else:
				y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

		self.optimizer.run(feed_dict={
			self.y_input : y_batch,
			self.action_input : action_batch,
			self.state_input : state_batch
			})
	

def main():
	env = gym.make(ENV_NAME)
	agent = DQN(env)

	for episode in xrange(EPISODE):
		state = env.reset()

		# train
		for step in xrange(STEP):
			action = agent.egreedy_action(state)
			next_state, reward, done, _ = env.step(action)

			if done:
				reward_agent = -1
			else:
				reward_agent = 0.1
			agent.perceive(state, action, reward, next_state, done)
			state = next_state
			if done:
				break

		# test
		if episode % 100 == 0:
			total_reward = 0
			for i in xrange(TEST):
				state = env.reset()
				for j in xrange(STEP):
					env.render()
					action = agent.action(state)
					state, reward, done, _ = env.step(action)
					total_reward += reward
					if done:
						break
			ave_reward = total_reward / TEST
			print 'episode: %d, Evaluation Average Reward: %f' % (episode, ave_reward)
			if ave_reward >= 200:
				break


if __name__ == '__main__':
	main()