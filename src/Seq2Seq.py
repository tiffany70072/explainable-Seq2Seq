""" The main part of the Seq2Seq model construction and training.
"""

import glob  # Search all matched files under a directory.
import keras.losses  # TODO: Remove this repeated import.
import numpy as np
import os  # For makedir.
import random
import sys

import data_reader
import utils

from keras.models import load_model  # TODO: Remove this repeated import.
from math import ceil
from numpy.random import seed
from pathlib import Path
from tensorflow.compat import v1 as tf
from tensorflow.compat.v1 import set_random_seed
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping

seed(1)
random.seed(30)
tf.disable_v2_behavior()
random_seed = 2
set_random_seed(random_seed)

"""
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
sess = tf.Session(config=config)
K.set_session(sess)
"""


class Seq2Seq(object):
	
	def __init__(self, mode, task, model_path, log_file=None, units=128, epochs=20):
		self.batch_size = 64
		self.epochs = epochs
		self.data = task 	# Default: "task1".
		self.units = units 	# Default: 128.
		self.model_path = model_path
		
		self.load_data(task)
		if mode == "train":
			self.logfile = log_file
			fout = open(self.logfile, "a")
			fout.write("+" * 50 + "\n")
			fout.write("data = %s\n" % self.data)
			fout.write("units = %.d\n" % self.units)
			fout.close()
		
	def load_data(self, task):
		self.encoder_in, self.decoder_in, self.decoder_out = data_reader.load_data(task = self.data)
		
		# TODO: change max_len = 10, larger than y.shape[1] 
		self.src_max_len = self.encoder_in.shape[1]
		self.tgt_max_len = self.decoder_out.shape[1]

		if task == "task1":
			self.src_token_size = np.max(self.encoder_in) + 1  
		elif task == "control_length":
			self.src_token_size = np.max(self.encoder_in) + 1  # TODO: Remove this if/else.
		
		self.tgt_token_size = np.max(self.decoder_out) + 1
		print("(Load data) token_size  =", self.src_token_size, self.tgt_token_size)

		self.cut_validation()
		self.target = np.zeros([self.batch_size, self.decoder_out.shape[1], self.tgt_token_size])

	def cut_validation(self):
		# TODO: cut training, validation and testing
		split_result = data_reader.data_split(self.encoder_in, self.decoder_in, self.decoder_out)
		self.encoder_in = split_result[0]
		self.decoder_in = split_result[1]
		self.decoder_out = split_result[2]
		self.encoder_in_valid = split_result[3]
		self.decoder_in_valid = split_result[4]
		self.decoder_out_valid = split_result[5]
		self.encoder_in_test = split_result[6]
		self.decoder_in_test = split_result[7]
		self.decoder_out_test = split_result[8]

		print("(Cut validation) training size:", self.encoder_in.shape)
		print("(Cut validation) validation size:", self.encoder_in_valid.shape)
		print("(Cut validation) testing size:", self.encoder_in_test.shape)

	def generate_batch_data(self, encoder_in, decoder_in, decoder_out):
		bs = self.batch_size
		data_num = decoder_out.shape[0]
		loopcount = data_num // bs
		sampling_container = np.ones((bs, 1))
		while True:
			i = random.randint(0, loopcount-1)
			batch_index = [j for j in range(i*bs, (i+1)*bs)]
			
			if random.uniform(0.0, 1.0) > 0.5:
				sampling_ratio = sampling_container
			else:
				sampling_ratio = sampling_container * 0
			yield [encoder_in[batch_index], decoder_in[batch_index], sampling_ratio], decoder_out[batch_index]
			# yield [encoder_in[batch_index], decoder_in[batch_index], np.array([0.5])], decoder_out[batch_index]
			# gc.collect()
	
	def load_seq2seq(self, model_path):
		# TODO: Combine this function to the one in utils.py.
		keras.losses.custom_loss = utils.masked_perplexity_loss
		self.seq2seq_model.load_weights(os.path.join(model_path, "seq2seq.h5")) 
		

	def save_seq2seq(self):
		"""Save trained Seq2Seq model."""

		# Check whether path exists.
		file_path = Path(self.model_path)
		if not file_path.exists():
			os.mkdir(self.model_path)
		self.seq2seq_model.save_weights(os.path.join(self.model_path, "seq2seq.h5")) 
		self.encoder_model.save_weights(os.path.join(self.model_path, "encoder.h5"))
		self.decoder_model.save_weights(os.path.join(self.model_path, "decoder.h5"))
	
	def quick_validation(N=100):
		pred = self.seq2seq_model.predict([self.encoder_in_test[:N], self.decoder_in_test[:N], np.zeros((N, 1))])
		utils.compute_accuracy(self.decoder_out_test[:N], pred)
		# TODO: check this function.

	def train(self): 
		"""Train Seq2Seq model."""

		print("\n\nStart training")
		print("(Training) train data =", self.encoder_in.shape, self.decoder_in.shape, self.decoder_out.shape)
		print("(Training) valid data =", self.encoder_in_valid.shape, self.decoder_in.shape, self.decoder_out.shape)

		# adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, clipvalue=5.0) # amsgrad=False, 
		self.seq2seq_model.compile(optimizer="adam",
								loss=utils.masked_perplexity_loss, 
								metrics=[metrics.categorical_accuracy, utils.reshape_accuracy, utils.accuracy_length])
		steps_per_epoch = ceil(self.decoder_out.shape[0] / self.batch_size)
		validation_steps = ceil(self.decoder_out_valid.shape[0] / self.batch_size)
		earlyStop = [EarlyStopping(monitor="val_reshape_accuracy", patience=1, verbose=2),]
		# quick_validation()

		for epoch in range(self.epochs):
			history = self.seq2seq_model.fit_generator(
				self.generate_batch_data(self.encoder_in, self.decoder_in, self.decoder_out),
				steps_per_epoch = steps_per_epoch,
				validation_data = self.generate_batch_data(self.encoder_in_valid, self.decoder_in_valid, self.decoder_out_valid),
				validation_steps = validation_steps, 
				callbacks = earlyStop)  # Default epochs = 1 in keras.

			with open(self.logfile, "a") as fout:
				fout.write("epochs = %d,\t" % epoch)
				fout.write("loss = %.3f,\t" % history.history["loss"][0])
				fout.write("acc = %.3f,\t" % history.history["reshape_accuracy"][0])
				fout.write("val_loss = %.3f,\t" % history.history["val_loss"][0])
				fout.write("val_acc = %.3f\n" % history.history["val_reshape_accuracy"][0])
			
			if history.history["val_loss"][0] == 1.0 and history.history["val_reshape_accuracy"][0] == 1.0: 
				break
			# quick_validation()

		self.save_seq2seq()

	def check_accuracy(self, check_list=["word"], N=100):
		"""Compute accuracy of different control signals."""
		# TODO: Support other signal: "word", "length", "rhyme", "POS".

		for check_item in check_list:
			if check_item == "word":
				pred = self.seq2seq_model.predict([self.encoder_in_test[:N], self.decoder_in_test[:N], np.zeros((N, 1))])
				accuracy_tensor = utils.reshape_accuracy(self.decoder_out_test[:N], pred)
				accuracy = np.mean(K.get_value(accuracy_tensor))
				print("Accuracy (word) = %.3f" % accuracy)

			elif check_item == "length":
				pred = self.seq2seq_model.predict([self.encoder_in_test[:N], self.decoder_in_test[:N], np.zeros((N, 1))])
				accuracy_tensor = utils.accuracy_length(self.decoder_out_test[:N], pred)
				print("Debug check shape:", K.get_value(accuracy_tensor).shape)
				accuracy = np.mean(K.get_value(accuracy_tensor))
				print("Accuracy (length) = %.3f" % accuracy)

				# print(np.argmax(pred[:10], -1))
				# print(self.encoder_in_test[:10])
				# print(self.decoder_in_test[:10])
				# print(self.decoder_out_test[:10])

			else:
				print("This item %s is not supported in accuracy check list." % check_item)
				continue


