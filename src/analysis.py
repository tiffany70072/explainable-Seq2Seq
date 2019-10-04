"""Anaylze the hidden state of the trained model."""

import data_reader
import numpy as np 
import pdb
import tensorflow.compat.v1 as tf
import tensorflow.keras.backend as K

from collections import defaultdict
from tensorflow.keras.models import Model
from tensorflow.keras.activations import hard_sigmoid
from tensorflow.keras.activations import tanh
	

def get_sample(task, N=100):
	"""Store samples based on their labels and time steps."""
	# TODO: Implement reading N from outside and only read N samples for each class.
	# TODO: Read src as the same way in other functions in this file.
	x, y, y_output = data_reader.load_data(task)  #, mode = 'analysis') 
	# TODO: Change variable's name.
	split_result = data_reader.data_split(x, y, y_output)  
	x_test = split_result[6]
	y_test = split_result[7]
	print('(Get sample) sample =', x_test.shape, y_test.shape)
		
	if task == 'task1':
		# Only 5 classes in this class.
		container = {0: [], 1: [], 2: [], 3: [], 4: []}  
		position = np.argmax(y, axis = -1)

		for i, p in enumerate(list(position)):
			if len(container[p]) < num: 
				container[p].append(i)
		print('container', [len(container[i]) for i in range(5)])

	elif task == "control_length":
		# Container (dict) stores sample index of each class. Map class to sample index.
		container = defaultdict(list)  
		# The token of [length = 1] is 8.
		basic_length = 3 + 5 - 1  
		for i, sample in enumerate(x_test):
			length = sample[-1] - basic_length
			if len(container[length]) < N:
				container[length].append(i)
		
		for key in container.keys():
			if len(container[key]) < N:
				print("Error: Samples for key %d is not enough (%d < N = %d)." % (key, len(container[key], N)))
				# TODO: Return error here, or remove this key.
		# pdb.set_trace()
		# Example: container.keys() = dict_keys([6, 5, 10, 1, 7, 4, 3, 2, 8, 9])

	print('(Get sample) Check container: ', [len(container[key]) for key in container.keys()])
	return container


def verify_state_correctness(one_gate_values, keras_h, keras_output, gate_values, hidden , output_tokens):
	"""Check if the gate values are correct."""
	# TODO: call this function in get_hidden_state.
	my_h = K.get_value(one_gate_values['h'])
	keras_h = keras_h[1]
	print(my_h.shape, my_h[0][:3], my_h[1][:3])
	print(keras_output.shape, keras_output[0][:3], keras_output[1][:3])
	print(keras_h.shape, keras_h[0][:3], keras_h[1][:3])
	for w in ['r', 'z', 'h', 'hh']: 
		gate_values[w][length, i] = K.get_value(one_gate_values[w])

	print('hidden =', len(hidden), hidden[0].shape, hidden[1].shape)  # Example: (1000, 1, 32), (1000, 32)
	print('output_tokens =', output_tokens.shape)  # Example: (1000, 1, 5)


def get_hidden_state(seq2seq, sample, SOS_token = 1):
	src = seq2seq.encoder_in_test

	dec_layer_model = Model(inputs = seq2seq.decoder_model.input, 
		outputs = seq2seq.decoder_model.get_layer('decoder_gru').get_output_at(-1))
		# outputs = seq2seq.decoder_model.get_layer('decoder_gru').output)

	# Dimension of hidden state = class, sample, length, dimension.
	# TODO: Handle if there is no any key.
	first_key = list(sample.keys())[0]  
	container = np.zeros([len(sample), len(sample[first_key]), seq2seq.tgt_max_len, seq2seq.units])
	print("(Gate value) container.shape =", container.shape)

	# Store each class in for loop.
	for key_index, key in enumerate(sorted(sample.keys())):
		batch_index = sample[key]
		print('(Gate_value) batch_index =', batch_index[:10])
		encoder_output = seq2seq.encoder_model.predict(src[batch_index])[0] 
		print('(Gate_value) encoder_output =', encoder_output.shape)

		decoder_states = encoder_output
		decoder_inputs = np.full([len(sample[key]), 1], SOS_token) # first token is SOS

		for t in range(seq2seq.tgt_max_len):
			hidden = dec_layer_model.predict([decoder_inputs] + [decoder_states], verbose = 0)
			# decoder_output = hidden[0].reshape([hidden[0].shape[0], hidden[0].shape[2]])
			# decoder_emb = np.squeeze(emb_layer_model.predict(decoder_inputs, verbose = 0), axis = 1)
			# one_gate_values = propagate_gru(weight, decoder_emb, decoder_states)
			output_tokens, decoder_states = seq2seq.decoder_model.predict([decoder_inputs] + [decoder_states], verbose = 0)
			sampled_token_index = np.argmax(output_tokens[:, -1, :], axis = -1) # Sample a token
			decoder_inputs[:, 0] = sampled_token_index[:]
			container[key_index, :, t:t+1, :] = hidden[0]
		# break
		print('container =', container.shape)

	return container


def POS_postprocess(gate_values, src, samples_idx, pos_dict):
	gate_values = {
		'r': pos_postprocess(gate_values['r'], src, samples_idx, pos_dict),
		'z': pos_postprocess(gate_values['z'], src, samples_idx, pos_dict),
		'h': pos_postprocess(gate_values['h'], src, samples_idx, pos_dict),
		'hh': pos_postprocess(gate_values['hh'], src, samples_idx, pos_dict), 
		}
	# shape = (10, 1000, 128).
	return gate_values


def get_gate_values(seq2seq, sample):
	"""Load gate values in GRU from trained Seq2Seq."""
	# TODO: Change to read variable values from other way.
	
	SOS_token = 1
	tgt_max_len = seq2seq.tgt_max_len  
	encoder_model = seq2seq.encoder_model
	decoder_model = seq2seq.decoder_model
	src = seq2seq.encoder_in_test
	samples_idx = sample  # TODO: remove this variable.
	units = seq2seq.units
	first_key = list(samples_idx.keys())[0]  # TODO: Handle if there is no any key.
	N = len(samples_idx[first_key])

	def propagate_gru(weight, inputs, states, units):
		kernel = K.variable(weight[0])  # shape = (input_dim, self.units * 3)
		recurrent_kernel = K.variable(weight[1])  # shape = (self.units, self.units * 3)
		bias = K.variable(weight[2])  # bias_shape = (3 * self.units,)
		
		# Update gate.
		kernel_z = kernel[:, :units]
		recurrent_kernel_z = recurrent_kernel[:, :units]
		# Reset gate.
		kernel_r = kernel[:, units:units * 2]
		recurrent_kernel_r = recurrent_kernel[:, units:units * 2]
		# New gate.
		kernel_h = kernel[:, units * 2:]
		recurrent_kernel_h = recurrent_kernel[:, units * 2:]

		# Assume use bias, not reset_after
		input_bias_z = bias[:units]
		input_bias_r = bias[units: units * 2]
		input_bias_h = bias[units * 2:]
		# Bias for hidden state - just for compatibility with CuDNN.

		# Call 
		inputs = K.variable(inputs)	 # Not sure.
		states = K.variable(states)	 # Not sure.
		h_tm1 = states  # Previous memory state.

		# Assume no dropout in this layer and self.implementation = 1 and not reset_after.
		inputs_z = inputs
		inputs_r = inputs
		inputs_h = inputs

		x_z = K.bias_add(K.dot(inputs_z, kernel_z), input_bias_z)
		x_r = K.bias_add(K.dot(inputs_r, kernel_r), input_bias_r)
		x_h = K.bias_add(K.dot(inputs_h, kernel_h), input_bias_h)
				   
		recurrent_z = K.dot(h_tm1, recurrent_kernel_z)
		recurrent_r = K.dot(h_tm1, recurrent_kernel_r)
					
		z = hard_sigmoid(x_z + recurrent_z)  # Recurrent activation = 'hard_sigmoid'.
		r = hard_sigmoid(x_r + recurrent_r)

		recurrent_h = K.dot(r * h_tm1, recurrent_kernel_h)
		hh = tanh(x_h + recurrent_h) 	# Activation = 'tanh'.	   
		# Previous and candidate state mixed by update gate.
		h = z * h_tm1 + (1 - z) * hh
		
		# for w in [r, z, h, hh]: 
			# w = K.get_value(w)
			# print(np.percentile(w, [0, 1, 25, 50, 75, 99, 100]))
		return {'r': r, 'z': z, 'h': h, 'hh': hh}
	
	weight = decoder_model.get_layer('decoder_gru').get_weights()
	# for j in range(len(weight)): 
	# print(j, weight[j].shape) # (128, 384), (128, 384), (384,)
	# encoder_model = Model(inputs=model.input, outputs=model.get_layer('forward').output)
	dec_layer_model = Model(inputs=decoder_model.input, 
		outputs=decoder_model.get_layer('decoder_gru').get_output_at(-1))
	emb_layer_model = Model(inputs=decoder_model.get_layer('decoder_emb').get_input_at(-1), 
		outputs=decoder_model.get_layer('decoder_emb').output)
	# emb_layer_model = Model(inputs=decoder_model.get_layer('decoder_emb').input, 
	#	outputs=decoder_model.get_layer('decoder_emb').output)
	
	gate_values = {'r': np.zeros([len(samples_idx), tgt_max_len, N, units]), 
				'z': np.zeros([len(samples_idx), tgt_max_len, N, units]),
				'h': np.zeros([len(samples_idx), tgt_max_len, N, units]),
				'hh': np.zeros([len(samples_idx), tgt_max_len, N, units])}
	# Example shape = (12, 13, 500, 128).
	print("(Gate value) Check shape =", gate_values['r'].shape)

	for key_index, key in enumerate(sorted(samples_idx.keys())):
		batch_index = samples_idx[key]
		print('(Gate_value) batch_index =', batch_index[:10])
		encoder_output = encoder_model.predict(src[batch_index])[0] 
		decoder_states = encoder_output
		decoder_inputs = np.full([len(samples_idx[key]), 1], SOS_token) # first token is SOS

		for i in range(tgt_max_len):
			# keras_h = dec_layer_model.predict([decoder_inputs] + [decoder_states], verbose = 0)
			# keras_output = keras_h[0].reshape([keras_h[0].shape[0], keras_h[0].shape[2]])

			decoder_emb = np.squeeze(emb_layer_model.predict(decoder_inputs, verbose=0), axis=1)
			one_gate_values = propagate_gru(weight, decoder_emb, decoder_states, units)
			output_tokens, decoder_states = decoder_model.predict([decoder_inputs] + [decoder_states], verbose=0)
			sampled_token_index = np.argmax(output_tokens[:, -1, :], axis=-1)  # Sample a token.
			decoder_inputs[:, 0] = sampled_token_index[:]
			
			for w in ['r', 'z', 'h', 'hh']: 
				gate_values[w][key_index, i] = K.get_value(one_gate_values[w])

	# TODO: Deal with POS task with different ways.
	if mode == 'pos':
		gate_values = pos_postprocess(gate_values, src, samples_idx, pos_dict)
		
	# pdb.set_trace()
	return gate_values


def get_dense_values(seq2seq, samples_idx):
	"""Get hidden values in last fully connected layer."""
	
	SOS_token = 1
	tgt_max_len = seq2seq.tgt_max_len
	encoder_model = seq2seq.encoder_model
	decoder_model = seq2seq.decoder_model
	src = seq2seq.encoder_in_test
	output_dim = seq2seq.tgt_token_size
	first_key = list(samples_idx.keys())[0]  # TODO: Handle if there is no any key.
	N = len(samples_idx[first_key])

	dense_values = np.zeros([len(samples_idx), tgt_max_len, N, output_dim])  # (12, 13, 500, 5000x)
	dense_layer = Model(inputs=decoder_model.input, 
		outputs=decoder_model.get_layer('output_dense').get_output_at(-1))
	
	for key_index, key in enumerate(sorted(samples_idx.keys())):
		batch_index = samples_idx[key]
		encoder_output = encoder_model.predict(src[batch_index], batch_size=256)[0] 
		decoder_states = encoder_output
		decoder_inputs = np.full([len(samples_idx[key]), 1], SOS_token)  # The first token is SOS.

		for i in range(tgt_max_len):
			one_dense_values = dense_layer.predict([decoder_inputs] + [decoder_states], verbose=0, batch_size=256)
			output_tokens, decoder_states = decoder_model.predict([decoder_inputs] + [decoder_states], verbose=0, batch_size=256)
			sampled_token_index = np.argmax(output_tokens[:, -1, :], axis=-1)  # Sample a token.
			decoder_inputs[:, 0] = sampled_token_index[:]
			dense_values[key_index, i] = np.copy(np.squeeze(one_dense_values, axis=1))  # Change shape: (500, 1, 5000x) -> (500, 5000x)

	return dense_values	


def output_weights(seq2seq):
	for layer in seq2seq.seq2seq_model.layers:
		# print("config =", layer.get_config())
		print("name =", layer.get_config()["name"])
		weights = layer.get_weights()
		print("len(weights) =", len(weights))
		for w in weights:
			print("w =", w.shape)


def get_gru_weights(seq2seq):
	# output_weights(seq2seq)
	
	container = {}
	for layer_name in ['encoder_emb', 'forward', 'decoder_emb', 'decoder_gru', 'output_dense']:
		weights = seq2seq.seq2seq_model.get_layer(layer_name).get_weights()
		print('\nlayer_name =', layer_name)
		container[layer_name] = weights

		continue

		for weight in weights:
			print("\nweight =", weight.shape)  # One vector or one 2-d array.
			if len(weight.shape) == 1:
				for w in weight:
					print("{:2.2f}".format(w), end=' ')
				# print(round(w) for w in weight)
				print()
			elif len(weight.shape) == 2:
				for i in range(weight.shape[0]):
					for j in range(weight.shape[1]):
						print("{:2.2f}".format(weight[i][j]), end=' ')
					print()
			else:
				print('No this kind of weight shape to output.')
		
	print('(Get weights) container =', container.keys())
	return container


def main():
	get_sample()


if __name__ == '__main__':
	main()