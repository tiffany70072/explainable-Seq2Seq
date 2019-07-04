import numpy as np
import tensorflow as tf
import keras.backend as K

import read_data

def write_results(results, output_file):
	print('output filename =', output_file)
	fout = open(output_file, 'w')
	for i in range(len(results)):
		fout.write(" ".join(results[i]))
		fout.write("\n")

def masked_perplexity_loss(y_true, y_pred):
	mask = K.all(K.equal(y_true, 0), axis=-1) # label padding as zero in y_true
	mask = 1 - K.cast(mask, K.floatx())
	nomask = K.sum(mask)
	loss = K.sparse_categorical_crossentropy(y_true, y_pred) * mask # multiply categorical_crossentropy with the mask
	#print (K.sparse_categorical_crossentropy(y_true, y_pred).shape)
	return tf.exp(K.sum(loss)/nomask)

def load_model(model_filename, model_path = 'saved_model'):
	from keras.models import load_model
	import keras.losses
	keras.losses.custom_loss = masked_perplexity_loss

	seq2seq_model = load_model(model_path+'/seq2seq_'+model_filename+'.h5', custom_objects={'masked_perplexity_loss': masked_perplexity_loss})
	encoder_model = load_model(model_path+'/encoder_'+model_filename+'.h5', custom_objects={'masked_perplexity_loss': masked_perplexity_loss})
	decoder_model = load_model(model_path+'/decoder_'+model_filename+'.h5', custom_objects={'masked_perplexity_loss': masked_perplexity_loss})
	print('load model finished\n\n')

	return seq2seq_model, encoder_model, decoder_model
	
def check_weights(model):
	for i, layer in enumerate(model.layers):
		print('layer', i)
		weight = layer.get_weights()
		print(len(weight), layer.input_shape, layer.output_shape)
		for j in range(len(weight)):
			print(j, weight[j].shape)
		
	'''
	0 (None, None) (None, None)
	0 (None, None) (None, None)
	1 (None, None) (None, None, 128)
	0 (50098, 128)
	3 (None, None, 128) [(None, 128), (None, 128)]
	0 (128, 384)
	1 (128, 384)
	2 (384,)
	3 [(None, None, 128), (None, 128)] [(None, None, 128), (None, 128)]
	0 (128, 384)
	1 (128, 384)
	2 (384,)
	2 (None, None, 128) (None, None, 50007)
	0 (128, 50007)
	1 (50007,)
	'''

def check_intermediate(model, src, tgt):
	from keras.models import Model

	batch_index = range(100)
	for layer_name in ['forward', 'decoder_gru', 'output_dense']:
		print('layer_name =', layer_name)
		intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
		intermediate_output = intermediate_layer_model.predict([src[batch_index], tgt[batch_index]])
		for j in range(len(intermediate_output)): print(j, intermediate_output[j].shape)

	'''
	layer_name = forward
	(100, 128)
	(100, 128)
	layer_name = decoder_gru
	(100, 13, 128)
	(100, 128)
	layer_name = output_dense
	(13, 50007) repeat 100 times???
	'''

def select_samples(src, num = 500, max_len = 12, mode = 'length', selected_len = 5, all_len = False): # all_len is for rhyme
	from read_data import load_vocabulary_from_pickle
	src_ctoi, src_itoc, tgt_ctoi, tgt_itoc = load_vocabulary_from_pickle()

	if mode == 'length':
		samples = [[] for i in range(max_len)] # store idx from length 1 to length 12
		count_finish = np.zeros([max_len])
		#assert np.min(src[:, -2]) == 50006 and np.max([src[:, -2]]) == 50020, 'vocab mapping error' # 50019? # check mapping
		for i, line in enumerate(src):
			length = int(line[-2]) - 6 + 1 # check vocab here
			if length > max_len: continue
			if count_finish[length-1] == 0 and len(samples[length-1]) < num: samples[length-1].append(i)
			else: count_finish[length-1] = 1
			if np.sum(count_finish) == len(samples): break
		for i in range(len(samples)): print(len(samples[i]), samples[i][-5:])
		samples = np.array(samples)
		samples_dict = {}
		for i in range(max_len): samples_dict[i] = i+1

	elif mode == 'rhyme':
		samples = []
		tmp_dict = {} # rhyme_idx to idx in samples_idx
		samples_len = []
		for i, line in enumerate(src):
			if int(line[-2]) - 6 + 1 == selected_len or (all_len == True and int(line[-2]) - 6 + 1 < 13):
				rhyme = line[-4]
				try:
					if len(samples[tmp_dict[rhyme]]) < num: 
						samples[tmp_dict[rhyme]].append(i)
						samples_len[tmp_dict[rhyme]].append(int(line[-2]) - 6 + 1)
				except KeyError:
					tmp_dict[rhyme] = len(samples)
					samples.append([i])
					samples_len.append([int(line[-2]) - 6 + 1])
		
		enough_idx = [i for i in range(len(samples)) if len(samples[i]) == num]
		samples = [i for i in samples if len(i) == num]
		samples_len = [i for i in samples_len if len(i) == num]
		#print('enough_idx:', enough_idx)
		#print('tmp_dict:', tmp_dict)
		tmp_dict = dict((v, k) for k, v in tmp_dict.items())
		#print('tmp_dict:', tmp_dict)
		samples_dict = {}
		for i in range(len(enough_idx)): samples_dict[i] = src_itoc[tmp_dict[enough_idx[i]]]
		#print('samples_dict:', samples_dict)
		#print('len(samples):', len(samples))
		#print('len(samples_len):', len(samples_len))
		samples = np.array(samples)
		samples_len = np.array(samples_len)
		#exit()

	elif mode == 'pos':
		samples = []
		for i in range(20):
			samples.append([i for i in range(i*num, (i+1)*num)])
			print(samples[i][:3])
		samples = np.array(samples)
		samples_dict = None
	
	#print('samples =', samples.shape)
	#print('samples_dict =', samples_dict)
	

	if all_len: return samples, samples_dict, samples_len
	return samples, samples_dict

def get_gate_values(encoder_model, decoder_model, src, samples_idx, pos_dict, mode = 'length', units = 128):
	#for layer in model.layers: print(str(layer))
	from keras.activations import tanh, hard_sigmoid
	from keras.models import Model

	def propagate_gru(weight, inputs, states, units = 128):
		kernel = K.variable(weight[0]) # shape=(input_dim, self.units * 3)
		recurrent_kernel = K.variable(weight[1]) # shape=(self.units, self.units * 3)
		bias = K.variable(weight[2]) # bias_shape = (3 * self.units,)
		# build weights
		# update gate
		kernel_z = kernel[:, :units]
		recurrent_kernel_z = recurrent_kernel[:, :units]
		# reset gate
		kernel_r = kernel[:, units:units * 2]
		recurrent_kernel_r = recurrent_kernel[:, units:units * 2]
		# new gate
		kernel_h = kernel[:, units * 2:]
		recurrent_kernel_h = recurrent_kernel[:, units * 2:]

		# assume use bias, not reset_after
		input_bias_z = bias[:units]
		input_bias_r = bias[units: units * 2]
		input_bias_h = bias[units * 2:]
		# bias for hidden state - just for compatibility with CuDNN

		# call 
		inputs = K.variable(inputs)	# not sure
		states = K.variable(states)	# not sure
		h_tm1 = states  # previous memory

		# assume no dropout in this layer and self.implementation = 1 and not reset_after
		inputs_z = inputs
		inputs_r = inputs
		inputs_h = inputs

		x_z = K.bias_add(K.dot(inputs_z, kernel_z), input_bias_z)
		x_r = K.bias_add(K.dot(inputs_r, kernel_r), input_bias_r)
		x_h = K.bias_add(K.dot(inputs_h, kernel_h), input_bias_h)
				   
		recurrent_z = K.dot(h_tm1, recurrent_kernel_z)
		recurrent_r = K.dot(h_tm1, recurrent_kernel_r)
					
		z = hard_sigmoid(x_z + recurrent_z) # recurrent activation = 'hard_sigmoid'
		r = hard_sigmoid(x_r + recurrent_r)

		recurrent_h = K.dot(r * h_tm1, recurrent_kernel_h)
		hh = tanh(x_h + recurrent_h) # activation = 'tanh'	   
		# previous and candidate state mixed by update gate
		h = z * h_tm1 + (1 - z) * hh
		
		#print(r.shape, z.shape, h.shape, hh.shape) # (100, 128) (100, 128) (100, 128) (100, 128)
		#[0.   0.   0.22  0.76 1.   1.   1. ]
		#[0.   0.   0.    0.33 1.   1.   1. ]
		#[-1.  -1.  -0.87 0.05 0.90 1.   1. ]
		#[-1.  -1.  -0.99 0.17 0.99 1.   1. ]
		#for w in [r, z, h, hh]: 
			#w = K.get_value(w)
			#print(np.percentile(w, [0, 1, 25, 50, 75, 99, 100]))
		return {'r': r, 'z': z, 'h': h, 'hh': hh}
	
	tgt_max_len = 13
	SOS_token = 1
	
	weight = decoder_model.get_layer('decoder_gru').get_weights()
	#for j in range(len(weight)): print(j, weight[j].shape) # (128, 384), (128, 384), (384,)
	#encoder_model = Model(inputs=model.input, outputs=model.get_layer('forward').output)
	dec_layer_model = Model(inputs=decoder_model.input, outputs=decoder_model.get_layer('decoder_gru').output)
	emb_layer_model = Model(inputs=decoder_model.get_layer('decoder_emb').input, outputs=decoder_model.get_layer('decoder_emb').output)
	gate_values = {'r': np.zeros([len(samples_idx), tgt_max_len, len(samples_idx[0]), units]), # (12, 13, 500, 128)
					'z': np.zeros([len(samples_idx), tgt_max_len, len(samples_idx[0]), units]),
					'h': np.zeros([len(samples_idx), tgt_max_len, len(samples_idx[0]), units]),
					'hh': np.zeros([len(samples_idx), tgt_max_len, len(samples_idx[0]), units])}

	for length in range(len(samples_idx)):
		batch_index = samples_idx[length]
		print('(gate_value) batch_index =', batch_index[:10])
		encoder_output = encoder_model.predict(src[batch_index])[0] 
		decoder_states = encoder_output
		decoder_inputs = np.full([len(samples_idx[length]), 1], SOS_token) # first token is SOS

		for i in range(tgt_max_len):
			#keras_h = dec_layer_model.predict([decoder_inputs] + [decoder_states], verbose = 0)
			#keras_output = keras_h[0].reshape([keras_h[0].shape[0], keras_h[0].shape[2]])

			decoder_emb = np.squeeze(emb_layer_model.predict(decoder_inputs, verbose = 0), axis = 1)
			one_gate_values = propagate_gru(weight, decoder_emb, decoder_states)
			output_tokens, decoder_states = decoder_model.predict([decoder_inputs] + [decoder_states], verbose = 0)
			sampled_token_index = np.argmax(output_tokens[:, -1, :], axis = -1) # Sample a token
			decoder_inputs[:, 0] = sampled_token_index[:]
			
			#my_h = K.get_value(one_gate_values['h'])
			#keras_h = keras_h[1]
			#print(my_h.shape, my_h[0][:3], my_h[1][:3])
			#print(keras_output.shape, keras_output[0][:3], keras_output[1][:3])
			#print(keras_h.shape, keras_h[0][:3], keras_h[1][:3])
			
			for w in ['r', 'z', 'h', 'hh']: gate_values[w][length, i] = K.get_value(one_gate_values[w])

	if mode == 'pos':
		gate_values = {
			'r': pos_postprocess(gate_values['r'], src, samples_idx, pos_dict),
			'z': pos_postprocess(gate_values['z'], src, samples_idx, pos_dict),
			'h': pos_postprocess(gate_values['h'], src, samples_idx, pos_dict),
			'hh': pos_postprocess(gate_values['hh'], src, samples_idx, pos_dict), # shape = (10, 1000, 128)
		}
		'''gate_values = {
			'r': pos_postprocess_steps(gate_values['r'], src, samples_idx, pos_dict),
			'z': pos_postprocess_steps(gate_values['z'], src, samples_idx, pos_dict),
			'h': pos_postprocess_steps(gate_values['h'], src, samples_idx, pos_dict),
			'hh': pos_postprocess_steps(gate_values['hh'], src, samples_idx, pos_dict), # shape = (10, 1000, 128)
		}'''
	return gate_values

	'''
	<keras.engine.topology.InputLayer object at 0x7f2dc00c66d8>
	<keras.engine.topology.InputLayer object at 0x7f2dc00bef28>
	<keras.layers.embeddings.Embedding object at 0x7f2dc0052048>
	<keras.layers.recurrent.GRU object at 0x7f2dc00521d0>
	<keras.layers.recurrent.GRU object at 0x7f2dc0052400>
	<keras.layers.core.Dense object at 0x7f2dc0052630>
	'''

def get_dense_weight(decoder_model):
	weight = decoder_model.get_layer('output_dense').get_weights()
	#print(len(weight), weight[0].shape)
	return weight

def get_embed_weight(encoder_model):
	weight = encoder_model.get_layer('encoder_emb').get_weights()
	weight = weight[0][6:6+13,:]
	return weight

def get_dense_values(encoder_model, decoder_model, src, samples_idx, output_dim):
	from keras.models import Model
	tgt_max_len = 13
	SOS_token = 1
	dense_values = np.zeros([len(samples_idx), tgt_max_len, len(samples_idx[0]), output_dim]) # (12, 13, 500, 5000x)
	#dense_layer = Model(inputs=decoder_model.input, outputs=decoder_model.get_layer('output_dense').output)
	dense_layer = Model(inputs=decoder_model.input, outputs=decoder_model.get_layer('output_dense').get_output_at(-1))
	
	for length in range(len(samples_idx)):
		#print(length)
		batch_index = samples_idx[length]
		encoder_output = encoder_model.predict(src[batch_index], batch_size = 256)[0] 
		decoder_states = encoder_output
		decoder_inputs = np.full([len(samples_idx[length]), 1], SOS_token) # first token is SOS

		for i in range(tgt_max_len):
			one_dense_values = dense_layer.predict([decoder_inputs] + [decoder_states], verbose = 0, batch_size = 256)

			output_tokens, decoder_states = decoder_model.predict([decoder_inputs] + [decoder_states], verbose = 0, batch_size = 256)
			sampled_token_index = np.argmax(output_tokens[:, -1, :], axis = -1) # Sample a token
			decoder_inputs[:, 0] = sampled_token_index[:]
			#if length == 0: print('one_dense_values =', one_dense_values.shape, one_dense_values[0][0][2])

			dense_values[length, i] = np.copy(np.squeeze(one_dense_values, axis = 1)) # (500, 1, 5000x) -> (500, 5000x)
	return dense_values	

def get_encoder_state(encoder_model, src, samples_idx, units = 128):
	#from keras.models import Model
	encoder_state = np.zeros([len(samples_idx), 500, 50, units]) 
	
	for length in range(len(samples_idx)):
		batch_index = samples_idx[length]
		encoder_output = encoder_model.predict(src[batch_index])[1] 
		encoder_state[length] = np.copy(encoder_output)
	return encoder_state

def count_common_pos(src):
	pos_count = {}
	EOS_token = 2
	NOP_token = 3
	for i in range(src.shape[0]):
		EOS_flag = 0
		for j in range(src.shape[1]):
			if EOS_flag == 0 and src[i][j] == EOS_token: EOS_flag = 1
			elif EOS_flag == 1 and src[i][j] != NOP_token:
				try: pos_count[src[i][j]] += 1
				except KeyError: pos_count[src[i][j]] = 0
			elif src[i][j] == NOP_token: break
	common_pos = sorted(pos_count, key = pos_count.get, reverse = True)
	top_N = common_pos[:10]
	print('top_N', top_N) 
	for i in top_N: print(pos_count[i])
	pos_dict = dict((c, i) for i, c in enumerate(top_N)) # chara
	print('pos_dict', pos_dict)
	return common_pos, pos_dict

def pos_postprocess(value, src, samples_idx, pos_dict, num = 10000):
	print('pos', value.shape)
	new_value = [[] for i in range(3)] # 3, num=1000, unit=128
	value = np.transpose(value, axes = [0, 2, 1, 3])
	value = np.reshape(value, [value.shape[0]*value.shape[1], value.shape[2], value.shape[3]])
	print('pos', value.shape)
	EOS_token = 2
	NOP_token = 3
	for i in range(samples_idx.shape[0]*samples_idx.shape[1]):
		EOS_flag = 0
		for j in range(src.shape[1]):
			if EOS_flag == 0 and src[i][j] == EOS_token: 
				EOS_flag = 1
				count = 0
			elif EOS_flag == 1 and src[i][j] != NOP_token:
				try: 
					pos_id = pos_dict[src[i][j]]
					if len(new_value[pos_id]) < num: new_value[pos_id].append(value[i][count])
					count += 1
				except KeyError: continue
			elif src[i][j] == NOP_token: break
			if EOS_flag == 1 and count >= 13: break
	for i in range(len(new_value)):
		print('len(new_value)', len(new_value[i]))
		if len(new_value[i]) > 0: print(len(new_value[i][0]))
	new_value = np.array(new_value)
	print('new value', new_value.shape)
	return new_value

def pos_postprocess_steps(value, src, samples_idx, pos_dict, num = 1000):
	print('pos', value.shape)
	new_value = [[ [] for _ in range(10) ] for _ in range(3)]#np.zeros([3, 10, num, 128]) # 3, step=10, num=1000, unit=128
	#print(len(new_value))
	#print(len(new_value[0]))
	#exit()
	value = np.transpose(value, axes = [0, 2, 1, 3]) # batch_num, step, num, unit
	value = np.reshape(value, [value.shape[0]*value.shape[1], value.shape[2], value.shape[3]])
	print('pos', value.shape)
	EOS_token = 2
	NOP_token = 3
	for idx in range(samples_idx.shape[0]*samples_idx.shape[1]):
		i = samples_idx[0][idx]
		EOS_flag = 0
		for j in range(src.shape[1]):
			if EOS_flag == 0 and src[i][j] == EOS_token: 
				EOS_flag = 1
				count = 0
			elif EOS_flag == 1 and src[i][j] != NOP_token:
				try: 
					pos_id = pos_dict[src[i][j]]
					try:
						if len(new_value[pos_id][count]) < num: 
							new_value[pos_id][count].append(value[idx][count])
					except IndexError:
						print(pos_id, count, num, len(new_value[pos_id][count]), len(new_value[pos_id]))
						exit()
					count += 1
				except KeyError: continue
			elif src[i][j] == NOP_token: break
			if EOS_flag == 1 and count >= 10: break
	for i in range(len(new_value)):
		print('len(new_value)', len(new_value[i]))
		if len(new_value[i]) > 0: print(len(new_value[i][0]))
	new_value = np.array(new_value)
	print('new value', new_value.shape)
	return new_value


def count_common_len(src):
	len_counter = {}
	for i, line in enumerate(src):
		length = int(line[-2]) - 6 + 1
		try: len_counter[length] += 1
		except KeyError: len_counter[length] = 1
	print('len_counter =', len_counter)
	# len_counter = {2: 3685, 3: 7469, 4: 11572, 5: 12241, 6: 10920, 7: 8692, 8: 5746, 9: 3570, 10: 2018, 11: 1093, 12: 595, 13: 334, 14: 195, 50015: 1870}

def main():
	#src, tgt = read_data.load_training_data()
	model, encoder, decoder = load_model('epoch10')
	src = read_data.load_testing_data()
	samples_idx = select_samples_by_length(src)

	#check_weights(model)
	#check_intermediate(model, src, tgt)
	#get_gate_values(model, decoder, src, samples_idx)
	
	#dense_weight = get_dense_weight(model) # [weight, bias] = [(128, 50006), (50006,)]
	#get_dense_values(model, decoder, src, samples_idx, dense_weight[0].shape[1])
	
	get_encoder_state(model, encoder, src, samples_idx)
	

if __name__ == '__main__':
	main()
