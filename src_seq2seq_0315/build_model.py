

#ERIC's code-----------------------------------------------------------------------------------------------------------------------------------------------------------------
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Input, Dropout, Concatenate, Dot, Embedding,Multiply,Add
from keras.layers import LSTM, GRU, SimpleRNN, Bidirectional
from keras.optimizers import Adam, RMSprop 
from keras import backend as K
import random
import tensorflow as tf
import numpy as np
from keras.layers import Lambda
from keras.initializers import RandomUniform
def slice(x,h1):
	return K.expand_dims(x[:,h1],1)
def where(x, ratio):
	teacher_forcing, x0, x1 = x[0], x[1], x[2]
	return tf.where(tf.greater(teacher_forcing, ratio), x0, x1)
def seq2seq(src_max_len, tgt_max_len, src_token_size, tgt_token_size, latent_dim = 128,teacher_forcing_ratio=0.5):
	rd = RandomUniform(minval=-0.08, maxval=0.08,seed=None)
	encoder_inputs = Input(shape=(None,),name='encoder_inputs')
	print('encoder_inputs =', encoder_inputs.shape)
	encoder_embedding = Embedding(src_token_size, latent_dim, embeddings_initializer=rd,input_length = None, mask_zero = True, name='encoder_emb')(encoder_inputs)
	print('encoder_embedding =', encoder_embedding.shape)
	encoder_time,encoder_state_h = GRU(latent_dim, kernel_initializer=rd,bias_initializer=rd,return_state=True,return_sequences=True, name='forward')(encoder_embedding)
	print ("encoder_state_h =",encoder_state_h.shape)
	encoder_model = Model(encoder_inputs,[encoder_state_h,encoder_time])

	decoder_inputs = Input(shape=(None,), name='decoder_inputs') 
	print('decoder_inputs =', decoder_inputs.shape)
	decoder_embedding = Embedding(tgt_token_size, latent_dim,embeddings_initializer=rd, input_length = None, name='decoder_emb')#(decoder_inputs)
	decoder_gru = GRU(latent_dim, kernel_initializer=rd,bias_initializer=rd,return_sequences=True, return_state=True, name = 'decoder_gru')
	decoder_dense = Dense(tgt_token_size, kernel_initializer=rd,bias_initializer=rd,activation='softmax', name='output_dense')

	inputs = Lambda(slice,arguments={'h1':0})(decoder_inputs)
	ret_dict = dict()
	HIDDEN_STATE = 'h_t'
	OUTPUT_STATE = 'o_t'
	SOFTMAX_STATE = 's_t'
	LENGTH = 'length'
	for i in [HIDDEN_STATE,OUTPUT_STATE,SOFTMAX_STATE,LENGTH]:
		ret_dict[i] = list()

	teacher_forcing = Input(shape=(None,),)
	
	for i in range(tgt_max_len):
		# Run the decoder on one timestep
		inputs_embed = decoder_embedding(inputs)
		decoder_outputs_time, state_h = decoder_gru(inputs_embed,initial_state=encoder_state_h)
		softmax = decoder_dense(decoder_outputs_time)
		outputs = Lambda(lambda x :K.argmax(x))(softmax)
		outputs = Lambda(lambda x :K.cast(outputs,'float32'))(outputs)
		decoder_inputs_time = Lambda(slice,arguments={'h1':i+1})(decoder_inputs)
		#inputs = Lambda(lambda x: K.switch(teacher_forcing>teacher_forcing_ratio,x[0],x[1]))([outputs,decoder_inputs_time])
		inputs = Lambda(where,arguments={'ratio':teacher_forcing_ratio})([teacher_forcing, outputs, decoder_inputs_time])
		encoder_state_h = state_h
		
		#ret_dict[HIDDEN_STATE] += [state_h]
		ret_dict[SOFTMAX_STATE] += [softmax]
		#ret_dict[OUTPUT_STATE] += [outputs]
		
	decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(ret_dict[SOFTMAX_STATE])
	#ERIC's code...........................................................................................

	# Define the model that will turn `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
	model = Model([encoder_inputs, decoder_inputs,teacher_forcing],decoder_outputs)
	#keras.losses.categorical_crossentropy(y_true, y_pred)
	#model.summary()
	
	
	decoder_state_input_h = Input(shape=(latent_dim,), name='decoder_input_h')
	decoder_outputs, state_h = decoder_gru(decoder_embedding(decoder_inputs), initial_state=decoder_state_input_h)
	print('decoder_outputs =', decoder_outputs)
	decoder_outputs = decoder_dense(decoder_outputs)
	print('decoder_outputs =', decoder_outputs)
	decoder_model = Model([decoder_inputs] + [decoder_state_input_h], [decoder_outputs] + [state_h])
	encoder_model.summary()
	decoder_model.summary()
	return model, encoder_model, decoder_model
'''def seq2seq(src_max_len, tgt_max_len, src_token_size, tgt_token_size, latent_dim = 128,teacher_forcing_ratio=0.5):
	rd = RandomUniform(minval=-0.08, maxval=0.08,seed=None)
	encoder_inputs = Input(shape=(None,),name='encoder_inputs')
	print('encoder_inputs =', encoder_inputs.shape)
	encoder_embedding = Embedding(src_token_size, latent_dim, embeddings_initializer=rd,input_length = None, mask_zero = True, name='encoder_emb')(encoder_inputs)
	print('encoder_embedding =', encoder_embedding.shape)
	encoder_time,encoder_state_h = GRU(latent_dim, kernel_initializer=rd,bias_initializer=rd,return_state=True,return_sequences=True, name='forward')(encoder_embedding)
	print ("encoder_state_h =",encoder_state_h.shape)
	encoder_model = Model(encoder_inputs,[encoder_state_h,encoder_time])

	decoder_inputs = Input(shape=(None,), name='decoder_inputs') 
	print('decoder_inputs =', decoder_inputs.shape)
	decoder_embedding = Embedding(tgt_token_size, latent_dim,embeddings_initializer=rd, input_length = None, name='decoder_emb')#(decoder_inputs)
	decoder_gru = GRU(latent_dim, kernel_initializer=rd,bias_initializer=rd,return_sequences=True, return_state=True, name = 'decoder_gru')
	decoder_dense = Dense(tgt_token_size, kernel_initializer=rd,bias_initializer=rd,activation='softmax', name='output_dense')

	inputs = Lambda(slice,arguments={'h1':0})(decoder_inputs)
	ret_dict = dict()
	HIDDEN_STATE = 'h_t'
	OUTPUT_STATE = 'o_t'
	SOFTMAX_STATE = 's_t'
	LENGTH = 'length'
	for i in [HIDDEN_STATE,OUTPUT_STATE,SOFTMAX_STATE,LENGTH]:
		ret_dict[i] = list()

	teacher_forcing = Input(shape=(None,),)
	
	for i in range(tgt_max_len):
		# Run the decoder on one timestep
		inputs_embed = decoder_embedding(inputs)
		decoder_outputs_time, state_h = decoder_gru(inputs_embed,initial_state=encoder_state_h)
		softmax = decoder_dense(decoder_outputs_time)
		outputs = Lambda(lambda x :K.argmax(x))(softmax)
		outputs = Lambda(lambda x :K.cast(outputs,'float32'))(outputs)
		decoder_inputs_time = Lambda(slice,arguments={'h1':i+1})(decoder_inputs)
		inputs = Lambda(lambda x: K.switch(teacher_forcing>teacher_forcing_ratio,x[0],x[1]))([outputs,decoder_inputs_time])
		
		encoder_state_h = state_h
		
		#ret_dict[HIDDEN_STATE] += [state_h]
		ret_dict[SOFTMAX_STATE] += [softmax]
		#ret_dict[OUTPUT_STATE] += [outputs]
		
	decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(ret_dict[SOFTMAX_STATE])
	#ERIC's code...........................................................................................

	# Define the model that will turn `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
	model = Model([encoder_inputs, decoder_inputs,teacher_forcing],decoder_outputs)
	#keras.losses.categorical_crossentropy(y_true, y_pred)
	model.summary()
	
	
	decoder_state_input_h = Input(shape=(latent_dim,), name='decoder_input_h')
	decoder_outputs, state_h = decoder_gru(decoder_embedding(decoder_inputs), initial_state=decoder_state_input_h)
	print('decoder_outputs =', decoder_outputs)
	decoder_outputs = decoder_dense(decoder_outputs)
	print('decoder_outputs =', decoder_outputs)
	decoder_model = Model([decoder_inputs] + [decoder_state_input_h], [decoder_outputs] + [state_h])
	encoder_model.summary()
	decoder_model.summary()
	return model, encoder_model, decoder_model
#ERIC's code-----------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
def enc_dec(src_max_len, tgt_max_len, src_token_size, tgt_token_size, latent_dim = 128):
	rd = RandomUniform(minval=-0.08, maxval=0.08,seed=None)
	encoder_inputs = Input(shape=(None,),name='encoder_inputs')
	#print('encoder_inputs =', encoder_inputs.shape)
	encoder_embedding = Embedding(src_token_size, latent_dim, embeddings_initializer=rd,input_length = None, mask_zero = True, name='encoder_emb')(encoder_inputs)
	#print('encoder_embedding =', encoder_embedding.shape)
	encoder_time,encoder_state_h = GRU(latent_dim, kernel_initializer=rd,bias_initializer=rd,return_state=True,return_sequences=True, name='forward')(encoder_embedding)
	#print ("encoder_state_h =",encoder_state_h.shape)
	encoder_model = Model(encoder_inputs,[encoder_state_h,encoder_time])

	decoder_inputs = Input(shape=(None,), name='decoder_inputs') 
	#print('decoder_inputs =', decoder_inputs.shape)
	decoder_embedding = Embedding(tgt_token_size, latent_dim,embeddings_initializer=rd, input_length = None, name='decoder_emb')#(decoder_inputs)
	decoder_gru = GRU(latent_dim, kernel_initializer=rd,bias_initializer=rd,return_sequences=True, return_state=True, name = 'decoder_gru')
	decoder_dense = Dense(tgt_token_size, kernel_initializer=rd,bias_initializer=rd,activation='softmax', name='output_dense')
	decoder_state_input_h = Input(shape=(latent_dim,), name='decoder_input_h')
	decoder_outputs, state_h = decoder_gru(decoder_embedding(decoder_inputs), initial_state=decoder_state_input_h)
	decoder_outputs = decoder_dense(decoder_outputs)
	#print('decoder_outputs =', decoder_outputs)
	decoder_model = Model([decoder_inputs] + [decoder_state_input_h], [decoder_outputs] + [state_h])
	return encoder_model,decoder_model
def seq2seq_backup(max_input_len, max_output_len, token_input_size, token_output_size, latent_dim = 128):
	#encoder_inputs = Input(shape=(None, token_input_size), name='encoder_inputs')
	encoder_inputs = Input(shape=(None,), name='encoder_inputs')
	encoder_embedding = Embedding(token_input_size, latent_dim, input_length = None, name='encoder_emb')(encoder_inputs)
	
	forward_h = GRU(latent_dim, return_state=True, name='forward')(encoder_embedding)[-1]
	backward_h = GRU(latent_dim, return_state=True, go_backwards=True, name='backward')(encoder_embedding)[-1]
	encoder_state_h = Concatenate(name='concat_h')([forward_h, backward_h])

	#decoder_inputs = Input(shape=(None, token_output_size), name='decoder_inputs') 
	decoder_inputs = Input(shape=(None,), name='decoder_inputs') 
	decoder_embedding = Embedding(token_output_size, latent_dim, input_length = None, name='decoder_emb')(decoder_inputs)
	decoder_gru = GRU(latent_dim * 2, return_sequences=True, return_state = True, name = 'decoder_gru')
	decoder_outputs, _ = decoder_gru(decoder_embedding, initial_state=encoder_state_h)
	#print('decoder_outputs =', decoder_outputs.shape)
	decoder_dense = Dense(token_output_size, activation='softmax', name='output_dense')
	decoder_outputs = decoder_dense(decoder_outputs)
	#print('decoder_outputs =', decoder_outputs.shape)

	# Define the model that will turn `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	#keras.losses.categorical_crossentropy(y_true, y_pred)
	model.summary()

	encoder_model = Model(encoder_inputs, encoder_state_h)
	decoder_state_input_h = Input(shape=(latent_dim*2,), name='decoder_input_h')
	decoder_outputs, state_h = decoder_gru(decoder_embedding, initial_state=decoder_state_input_h)
	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model = Model([decoder_inputs] + [decoder_state_input_h], [decoder_outputs] + [state_h])
	
	#encoder_model.summary()
	#decoder_model.summary()
	exit()
	return model, encoder_model, decoder_model

