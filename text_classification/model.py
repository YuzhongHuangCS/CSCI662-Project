import logging
import math
import os
import pickle
import random
from collections import Counter
import time
import h5py
import numpy as np
import tensorflow as tf
import pdb

logging.basicConfig(level=logging.DEBUG,
					format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
					datefmt="%Y-%m-%d %H:%M:%S")


class Model(object):
	def __init__(self, args):
		super(Model, self).__init__()
		self.args = args
		self.log = logging.getLogger(__name__)
		self.word2idx = None
		self.label2idx = None
		self.idx2word = None
		self.idx2label = None
		self.f = {}
		self.special_symbol = {
			'unknown': 0,
			'padding': 1,
		}

	@staticmethod
	def inverse_dict(d):
		return {v: k for k, v in d.items()}

	def build_vocabulary(self):
		vocab_filename = '{}{}vocab.pickle'.format(self.args.prefix, self.args.tag)
		if False and os.path.isfile(vocab_filename):
			self.log.debug('Loading from: %s', vocab_filename)
			with open(vocab_filename, 'rb') as f:
				self.word2idx, self.label2idx = pickle.load(f)
				self.idx2word = self.inverse_dict(self.word2idx)
				self.idx2label = self.inverse_dict(self.label2idx)
		else:
			word_counter = Counter()
			label_counter = Counter()
			for line in open(self.args.prefix + self.args.dataset[0], 'r', encoding='utf8'):
				label, sentence = line.strip().split('\t', 1)
				word_counter.update(sentence.split())
				label_counter.update([label])

			word2idx = {}
			for index, value in enumerate(word_counter.most_common()):
				word2idx[value[0]] = index + len(self.special_symbol)

			label2idx = {}
			for index, value in enumerate(label_counter.most_common()):
				label2idx[value[0]] = index

			self.log.debug('Saving to: %s', vocab_filename)
			with open(vocab_filename, 'wb') as f:
				pickle.dump([word2idx, label2idx], f, pickle.HIGHEST_PROTOCOL)

			self.word2idx = word2idx
			self.idx2word = self.inverse_dict(word2idx)
			self.label2idx = label2idx
			self.idx2label = self.inverse_dict(label2idx)

	def init_hdf5file(self, filename):
		hdf5_filename = filename + '.h5'
		if False and os.path.isfile(hdf5_filename):
			self.log.debug('Loading from: %s', hdf5_filename)
			self.f[filename] = h5py.File(hdf5_filename, 'r', libver='latest')
		else:
			self.log.debug('Saving to: %s', hdf5_filename)
			fd = h5py.File(hdf5_filename, 'w', libver='latest')
			self.f[filename] = fd

			fd.create_dataset('labels', (0,), dtype='int32', chunks=(self.args.line,), maxshape=(None,),
							  compression='gzip',
							  compression_opts=9)
			fd.create_dataset('features', (0, self.args.length), dtype='int32',
							  chunks=(self.args.line, self.args.length),
							  maxshape=(None, self.args.length), compression='gzip', compression_opts=9)
			self.write_hdf5file(filename)

	def read_textfile(self, filename):
		with open(filename, 'r', encoding='utf-8') as f:
			while True:
				features = []
				labels = []

				for i in range(self.args.line):
					line = f.readline()
					if line == '':
						break

					label, sentence = line.strip().split('\t', 1)
					feature = []
					for word in sentence.split():
						emb = self.word2idx.get(word, None)
						if emb is not None:
							feature.append(emb)

					feature = np.asarray(feature)
					if len(feature) < self.args.length:
						feature = np.pad(feature, (0, self.args.length - len(feature)), mode='constant',
										 constant_values=self.special_symbol['padding'])
					else:
						feature = feature[:self.args.length]
					features.append(feature)
					labels.append(self.label2idx[label])

				if len(labels) == 0:
					break
				else:
					yield [np.asarray(features), np.asarray(labels)]

	def write_hdf5file(self, filename):
		ds_f = self.f[filename]['features']
		ds_l = self.f[filename]['labels']

		for features, labels in self.read_textfile(filename):
			idx = len(ds_f)
			ds_f.resize(idx + len(features), axis=0)
			ds_f[idx:, :] = features
			ds_l.resize(idx + len(labels), axis=0)
			ds_l[idx:] = labels

	def read_hdf5file(self, filename, shuffle=False):
		ds_f = self.f[filename]['features']
		ds_l = self.f[filename]['labels']
		idx = len(ds_f)
		i_ary = list(range(0, idx, self.args.line))
		if shuffle:
			random.shuffle(i_ary)
		for i in i_ary:
			yield [ds_f[i:i + self.args.line, :], ds_l[i:i + self.args.line]]

	@staticmethod
	def mean_transform(features):
		return tf.reduce_mean(features, axis=1)

	def sgd(self):
		word_vocab_size = len(self.word2idx) + len(self.special_symbol)
		label_vocab_size = len(self.label2idx)

		features = tf.placeholder(tf.int32, shape=(None, self.args.length))
		labels = tf.placeholder(tf.int32, shape=(None,))

		if self.args.embedding != '':
			initial_embedding = np.zeros((word_vocab_size, self.args.dimension), dtype=np.float32)
			f = open(self.args.embedding, 'r', encoding='utf-8')
			self.log.info('Declared Embedding size: %s', f.readline())
			for line in f:
				parts = line.strip().split(' ')
				if parts[0] in self.word2idx:
					initial_embedding[self.word2idx[parts[0]]] = np.asarray([np.float32(t) for t in parts[1:]])
		else:
			initial_embedding = tf.random_uniform([word_vocab_size, self.args.dimension], -math.sqrt(3), math.sqrt(3))

		word_embedding = tf.Variable(initial_embedding, name="Embedding")
		embedded_features = tf.nn.embedding_lookup(word_embedding, features)
		#pdb.set_trace()
		W = tf.get_variable('WOutput', shape=(self.args.dimension, 20))
		embedded_features = tf.tensordot(embedded_features, W, axes=1)
		hidden_mean = self.mean_transform(embedded_features)
		#pdb.set_trace()
		dense_mean = tf.layers.dense(inputs=hidden_mean, units=label_vocab_size, name="Output",
									 kernel_initializer=tf.glorot_uniform_initializer())
		predicted_mean = tf.argmax(input=dense_mean, axis=1)
		loss_mean = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=dense_mean)

		var_embedding = [x for x in tf.trainable_variables() if 'Embedding' in x.name]
		#var_filter = [x for x in tf.trainable_variables() if 'filter' in x.name]
		var_output = [x for x in tf.trainable_variables() if 'Output' in x.name]

		lr = tf.Variable(self.args.lr_mean, trainable=False)
		lr_decay_op = lr.assign(lr * 0.97)
		train_mean = tf.train.AdamOptimizer(self.args.lr_mean).minimize(loss=loss_mean,
																		var_list=var_output + var_embedding)
		'''
		train_pca = [
			tf.train.AdamOptimizer(self.args.lr_pca).minimize(loss=self.pca_losses[i], var_list=[var_filter[i]]) for i
			in range(self.count)]
		train_output = tf.train.AdamOptimizer(self.args.lr_output).minimize(loss=loss_full, var_list=var_output)
		'''
		saver = tf.train.Saver()
		t = []
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			self.log.info('Start Training')
			# writer = tf.summary.FileWriter('logs', sess.graph)
			accuracies = []
			# train mean
			times = []
			for epoch in range(self.args.epoch_mean):
				trunk = 0
				for train_features, train_labels in self.read_hdf5file(self.args.prefix + self.args.dataset[0], True):
					t1 = time.time()
					_, train_loss = sess.run([train_mean, loss_mean],
											 feed_dict={features: train_features, labels: train_labels})
					t2 = time.time()
					times.append(t2-t1)
					self.log.debug('Epoch: {}, Trunk: {}, Train feature shape: {}, Loss: {}'.format(epoch, trunk,
																									train_features.shape,
																									train_loss))
					trunk += 1

				total = 0
				diff = 0
				for test_features, test_labels in self.read_hdf5file(self.args.prefix + self.args.dataset[1]):
					predicted_labels = sess.run(predicted_mean,
												feed_dict={features: test_features, labels: test_labels})
					total += len(test_labels)
					diff += np.count_nonzero(test_labels - predicted_labels)

				accuracy = 1 - diff / total
				accuracies.append(accuracy)
				sess.run(lr_decay_op)
				print('Apply lr decay, new lr: %f' % sess.run(lr))
				self.log.info('Training Mean: Epoch: {}, Test Accuracy: {}'.format(epoch, accuracy))

			print(np.sum(times))
			pdb.set_trace()

			# writer.close()
			save_path = saver.save(sess, self.args.prefix + 'model/model.ckpt')
			self.log.info("Model saved in path: {}".format(save_path))
			max_accuracy = np.max(accuracies)
			max_index = np.argmax(accuracies)
			self.log.info('Max accuracy: {}@{} epoch'.format(max_accuracy, max_index))

	def close(self):
		for fd in self.f.values():
			fd.close()
