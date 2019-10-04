"""Read training data and do some preprocessing."""


import numpy as np 
import os

from sklearn.model_selection import train_test_split


def insert_SOS(seq, SOS_token = 1):
	SOS_array = np.full([seq.shape[0], 1], SOS_token)
	seq = np.concatenate([np.full([seq.shape[0], 1], SOS_token), seq], axis = 1)
	print('seq after SOS inserting: ', seq[0], seq.shape)
	return seq


def insert_EOS(seq, EOS_token = 2):
	seq = np.concatenate([seq, np.full([seq.shape[0], 1], EOS_token)], axis = 1)
	print('seq after EOS inserting: ', seq[0], seq.shape)
	return seq


def insert_one_token(seq, token, insert_front=True):
	if insert_front: 
		seq = np.concatenate([np.full([seq.shape[0], 1], token), seq], axis = 1)
	else:  # Insert in the end.
		seq = np.concatenate([seq, np.full([seq.shape[0], 1], token)], axis = 1)
	print('Seq after inserting: ', seq[0], seq.shape)
	return seq


def load_data(task='task1', mode='train'):
	# TODO: implment reading task1 and task2 (autoencoder).
	
	if task == 'control_length':
		x = np.load(os.path.join('../data', task, 'in.npy'))
		y = np.load(os.path.join('../data', task, 'out.npy'))
		encoder_input = x[:]
		decoder_input = y[:]
		decoder_output = y[:, 1:]  # Remove SOS and add a PAD in the end.
		decoder_output = insert_one_token(decoder_output, 0, insert_front=False)
		decoder_output = np.reshape(decoder_output, [decoder_output.shape[0], decoder_output.shape[1], 1]) 
		return encoder_input, decoder_input, decoder_output

	
def data_split(x, y, z, test_size = 0.3, valid_size = 0.1):
	"""Call sklearn to do train test split."""

	x, x_test, y, y_test, z, z_test = train_test_split(x, y, z, test_size = 0.3, random_state = 777)
	x, x_valid, y, y_valid, z, z_valid = train_test_split(x, y, z, test_size = 0.1, random_state = 777)

	# print('(Data split) training size:', x.shape)
	# print('(Data split) validation size:', x_valid.shape)
	# print('(Data split) testing size:', x_test.shape)
	return [x, y, z, x_valid, y_valid, z_valid, x_test, y_test, z_test]

