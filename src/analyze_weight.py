"""Analyze the weights from trained model."""


import numpy as np
import pdb

def test_emb(emb_weight, one_hot):
	"""Test two embedding layers."""
	# emb_weight.shape should be (one_hot_size, hidden).
	for i in one_hot:
		print('label = %d, emb =' % i, emb_weight[i])


def test_dense(weight, bias, tgt_token_size=8):
	"""Test the fully connected layer."""
	# Prove the correlation of dense weight and hidden values.
	cases = [[1, -1, -1, -1], 
			[1, 1, 1, 1], 
			[1, -0.2, 0.35, -0.9], 
			[1, 0.4, 0.6, 0.75], 
			[0.99, -0.22,  0.33, -0.86], 
			[0.99, 0.46, 0.65, 0.84]]

	for case in cases:
		for i in range(tgt_token_size):
			print("%d, %.2f" % (i, sum(case * weight[:, i]) + bias[i]))


def hard_sigmoid(x):  
	"""Implement the numpy version of hard sigmoid from tensor version in tensorflow."""
	x = 0.2 * x
	x = x + 0.5
	x = np.clip(x, 0., 1.)
	return x


def tanh(x):
	"""Implement the numpy version of tanh from tensor version in tensorflow."""
	return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def test_decoder(W, U, b, units=4):
	"""Test the decoder GRU layer."""

	# Set weights.
	W_z, W_r, W_h = W[:, :units], W[:, units:2*units], W[:, 2*units:]
	U_z, U_r, U_h = U[:, :units], U[:, units:2*units], U[:, 2*units:]
	b_z, b_r, b_h = b[:units], b[units:2*units], b[2*units:]
	print("(Analyze weight) gru.shape =", W_z.shape, W_r.shape, W_h.shape, b_z.shape, b_r.shape, b_h.shape)
	# pdb.set_trace()

	# Set hidden values.
	h = [0.99, -0.22,  0.33, -0.86]
	expected_h = [0.99, 0.46, 0.65, 0.84]
	expected_z = [0.56, 0.45, 0.53, 0.0]
	expected_r = [1.0, 0.7, 1.0, 1.0]
	expected_hh = [0.99, 0.99, 0.99, 0.84]

	# Forward GRU.	
	z = hard_sigmoid(np.dot(h, U_z) + b_z)
	r = hard_sigmoid(np.dot(h, U_r) + b_r)
	recurrent_h = np.dot(np.multiply(r, h), U_h)
	print("recurrent_h =", recurrent_h)
	hh = tanh(recurrent_h + b_h)
	h = z * h + (1 - z) * hh

	# Output results.
	for gate, value in zip(['h', 'z', 'r', 'hh'], [h, z, r, hh]):
		print(gate, value)
	pdb.set_trace()
	return 


def main(weights, gate_values):
	length = 8
	for gate in ['h', 'z', 'r', 'hh']:
		print("Gate", gate)
		for time in range(10):
			print(time, end=" ")
			print(np.mean(gate_values[gate][length-1, time, :, :], axis=0))

	return 
	for layer_name, weight in weights.items():
		print('(Analyze weights) layer_name =', layer_name)
		if layer_name == 'encoder_emb':
			one_hot = range(18)
			test_emb(weight[0], one_hot)

		if layer_name == 'decoder_emb':
			one_hot = range(8)
			test_emb(weight[0], one_hot)
		
		if layer_name == 'output_dense':
			test_dense(weight[0], weight[1])
		
		if layer_name == 'decoder_gru':
			test_decoder(weight[0], weight[1], weight[2])
