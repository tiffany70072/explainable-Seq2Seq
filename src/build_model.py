import numpy as np
import random
import tensorflow.compat.v1 as tf

# Original import path: from tf.keras.initializers import RandomUniform
from tensorflow.compat.v1.keras.initializers import RandomUniform
from tensorflow.compat.v1.keras.layers import Dense, Embedding, Input, Lambda
from tensorflow.compat.v1.keras.layers import GRU 
# from tensorflow.compat.v1.keras.layers import Activation, Add, Bidirectional, Concatenate, Dot, Dropout, Multiply, SimpleRNN, LSTM
from tensorflow.compat.v1.keras.optimizers import Adam, RMSprop 
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential


def slice(x, h1):
	return K.expand_dims(x[:, h1], 1)


def where(x, ratio):
	teacher_forcing, x0, x1 = x[0], x[1], x[2]
	comparison = tf.greater(teacher_forcing, ratio) 
	return tf.where(comparison, x0, x1)


def seq2seq(src_max_len, tgt_max_len, src_token_size, tgt_token_size, latent_dim=128, teacher_forcing_ratio=0.5):
	rd = RandomUniform(minval=-0.08, maxval=0.08, seed=None)

	encoder_inputs = Input(shape=(None,), name='encoder_inputs')
	print('(Build model) encoder_inputs =', encoder_inputs.shape)
	encoder_embedding = Embedding(src_token_size, latent_dim, embeddings_initializer=rd, input_length=None, mask_zero=True, name='encoder_emb')(encoder_inputs)
	print('(Build model) encoder_embedding =', encoder_embedding.shape)
	encoder_time, encoder_state_h = GRU(latent_dim, kernel_initializer=rd, bias_initializer=rd, return_state=True, return_sequences=True, name='forward')(encoder_embedding)
	print ("(Build model) encoder_state_h =", encoder_state_h.shape)
	encoder_model = Model(encoder_inputs, [encoder_state_h, encoder_time])

	decoder_inputs = Input(shape=(None,), name='decoder_inputs') 
	print('(Build model) decoder_inputs =', decoder_inputs.shape)
	decoder_embedding = Embedding(tgt_token_size, latent_dim, embeddings_initializer=rd, input_length=None, name='decoder_emb')
	decoder_gru = GRU(latent_dim, kernel_initializer=rd, bias_initializer=rd, return_sequences=True, return_state=True, name='decoder_gru')
	decoder_dense = Dense(tgt_token_size, kernel_initializer=rd, bias_initializer=rd, activation='softmax', name='output_dense')

	inputs = Lambda(slice, arguments={'h1': 0})(decoder_inputs)
	softmax_state = []
	teacher_forcing = Input(shape=(None,),)
	decoder_state_h = encoder_state_h
	
	# Run decoder on each timestep.
	for i in range(tgt_max_len):
		inputs_embed = decoder_embedding(inputs)
		decoder_outputs_time, state_h=decoder_gru(inputs_embed, initial_state = decoder_state_h)
		softmax = decoder_dense(decoder_outputs_time)
		outputs = Lambda(lambda x: K.argmax(x))(softmax)
		outputs = Lambda(lambda x: K.cast(outputs, 'float32'))(outputs)
		decoder_inputs_time = Lambda(slice, arguments={'h1': i + 1})(decoder_inputs)
		inputs = Lambda(where, arguments={'ratio': teacher_forcing_ratio})([teacher_forcing, decoder_inputs_time, outputs])
		# inputs = Lambda(where, arguments={'ratio': 0.5})([teacher_forcing, outputs, decoder_inputs_time])
		
		decoder_state_h = state_h
		softmax_state += [softmax]
	decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(softmax_state)
	
	# Define the model that will turn "encoder_input_data" & "decoder_input_data" into "decoder_target_data".
	model = Model([encoder_inputs, decoder_inputs, teacher_forcing], decoder_outputs)
	# model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	# model.summary()
	
	decoder_state_input_h = Input(shape=(latent_dim,), name='decoder_input_h')
	decoder_outputs, state_h = decoder_gru(decoder_embedding(decoder_inputs), initial_state=decoder_state_input_h)
	print('(Build model) decoder_outputs =', decoder_outputs)
	decoder_outputs = decoder_dense(decoder_outputs)
	print('(Build model) decoder_outputs =', decoder_outputs)
	decoder_model = Model([decoder_inputs] + [decoder_state_input_h], [decoder_outputs] + [state_h])
	encoder_model.summary()
	decoder_model.summary()
	return model, encoder_model, decoder_model


def enc_dec(src_max_len, tgt_max_len, src_token_size, tgt_token_size, latent_dim=128):
	"""Get the empty encoder and decoder."""

	rd = RandomUniform(minval=-0.08, maxval=0.08,seed=None)
	encoder_inputs = Input(shape=(None,),name='encoder_inputs')
	encoder_embedding = Embedding(src_token_size, latent_dim, embeddings_initializer=rd, input_length=None, mask_zero=True, name='encoder_emb')(encoder_inputs)
	encoder_time,encoder_state_h = GRU(latent_dim, kernel_initializer=rd, bias_initializer=rd, return_state=True,return_sequences=True, name='forward')(encoder_embedding)
	encoder_model = Model(encoder_inputs,[encoder_state_h,encoder_time])

	decoder_inputs = Input(shape=(None,), name='decoder_inputs') 
	decoder_embedding = Embedding(tgt_token_size, latent_dim, embeddings_initializer=rd, input_length=None, name='decoder_emb')#(decoder_inputs)
	decoder_gru = GRU(latent_dim, kernel_initializer=rd, bias_initializer=rd, return_sequences=True, return_state=True, name='decoder_gru')
	decoder_dense = Dense(tgt_token_size, kernel_initializer=rd, bias_initializer=rd, activation='softmax', name='output_dense')
	decoder_state_input_h = Input(shape=(latent_dim,), name='decoder_input_h')
	decoder_outputs, state_h = decoder_gru(decoder_embedding(decoder_inputs), initial_state=decoder_state_input_h)
	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model = Model([decoder_inputs] + [decoder_state_input_h], [decoder_outputs] + [state_h])

	return encoder_model, decoder_model
