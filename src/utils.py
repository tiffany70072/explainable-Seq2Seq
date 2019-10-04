import numpy as np
import keras.losses
import tensorflow.compat.v1 as tf
import tensorflow.keras.backend as K

from keras.models import load_model


def masked_perplexity_loss(y_true, y_pred, PAD_token=0):
	"""Construct customer masked perplexity loss."""
	mask = K.all(K.equal(y_true, PAD_token), axis=-1)  # Label padding as zero in y_true
	mask = 1 - K.cast(mask, K.floatx())
	nomask = K.sum(mask)
	loss = K.sparse_categorical_crossentropy(y_true, y_pred) * mask  # Multiply categorical_crossentropy with the mask
	return tf.exp(K.sum(loss)/nomask)


def load_model(model_filename, model_path='saved_model'):
	keras.losses.custom_loss = masked_perplexity_loss
	seq2seq_model = load_model(model_path+'/seq2seq_'+model_filename+'.h5', custom_objects={'masked_perplexity_loss': masked_perplexity_loss})
	encoder_model = load_model(model_path+'/encoder_'+model_filename+'.h5', custom_objects={'masked_perplexity_loss': masked_perplexity_loss})
	decoder_model = load_model(model_path+'/decoder_'+model_filename+'.h5', custom_objects={'masked_perplexity_loss': masked_perplexity_loss})
	print('load model finished\n\n')

	return seq2seq_model, encoder_model, decoder_model


def reshape_accuracy(real, pred):
	# Use in training metrics.
	pred = K.cast(K.argmax(pred, axis = -1), K.floatx())
	real = K.cast(K.squeeze(real, axis = -1), K.floatx())
	# print('pred =', pred[:10])
	# print('real =', real[:10])

	# accuracy = K.mean(pred == real)
	# accuracy = tf.metrics.Accuracy(real, pred) # keras.
	# For tensorflow metrics.
	accuracy_tensor = K.cast(K.equal(real, pred), K.floatx())  # shape = (N, max_length)
	

	return accuracy_tensor  


def accuracy_length(y_true, y_pred):
	"""Compute the length control signal accuracy by matching position of EOS token."""

	EOS_token = 2
	
	mask_PAD = K.all(K.equal(y_true, 0), axis=-1)  # Shape = (N, max_length, 1)
	mask_PAD = 1 - K.cast(mask_PAD, K.floatx())
	mask_PAD = tf.squeeze(mask_PAD)  # Shape = (N, max_length). False if this is PAD.

	y_pred = K.cast(K.argmax(y_pred, axis=-1), K.floatx())
	y_pred = y_pred * mask_PAD  # Shape = (N, max_length)

	filter_EOS = K.all(K.equal(y_true, EOS_token), axis=-1)
	filter_EOS = K.cast(filter_EOS, K.floatx())
	filter_EOS = tf.squeeze(filter_EOS)  # Shape = (N, max_length), True if it is EOS.

	y_expected = K.equal(y_pred * filter_EOS, float(EOS_token))
	y_expected = K.cast(y_expected, K.floatx())  # Shape = (N, max_length)
	y_expected = K.sum(y_expected, axis=-1)  # Shape = (N, )
	y_expected = K.cast((K.equal(y_expected, 1.0)), K.floatx())
	y_result = K.cast(K.equal(y_pred, float(EOS_token)), K.floatx())  # Shape = (N, max_length)
	y_result = K.sum(y_result, axis=-1)  # Shape = (N, )
	y_result = K.cast((K.equal(y_result, 1.0)), K.floatx())

	accuracy = y_expected * y_result  # Shape = (N, )
	accuracy = (K.sum(accuracy) / K.sum(filter_EOS))  # / K.sum(tf.ones_like(accu)))

	return accuracy


def write_results(results, output_file):
	print('output filename =', output_file)
	fout = open(output_file, 'w')
	for i in range(len(results)):
		fout.write(" ".join(results[i]))
		fout.write("\n")


