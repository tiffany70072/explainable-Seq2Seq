import numpy as np 
import sys
import pdb
import gc
from random import randint
import os

import read_data
import build_model
import utils
from utils import masked_perplexity_loss

from keras.utils.np_utils import to_categorical
from numpy.random import seed
from tensorflow import set_random_seed
seed(1)
set_random_seed(2)

import tensorflow as tf
import keras.backend as K
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
sess = tf.Session(config=config)
K.set_session(sess)

def set_decoder_output_data(decoder_input):
	decoder_output = decoder_input.copy()
	for i in range(len(decoder_output)): 
		decoder_output[i, :-1] = decoder_input[i, 1:] # remove the first token in decoder output
		decoder_output[i, -1] *= 0
	decoder_output = np.reshape(decoder_output, [decoder_output.shape[0], decoder_output.shape[1], 1])
	return decoder_output

#def categorical_3d(x, max_len, token_size):
#	y = np.zeros([x.shape[0], x.shape[1], token_size])
#	for i in range(y.shape[0]): y[i] = to_categorical(x[i], num_classes = token_size)
#	return y


class LyricGenerator(object):
	def __init__(self, mode = 'train'):
		self.batch_size = 64
		self.epochs = 20
		#self.N = 600000

	def load_data(self):
		self.src_ctoi, self.src_itoc, self.tgt_ctoi, self.tgt_itoc = read_data.load_vocabulary_from_pickle()
		src_padded, tgt_padded = read_data.load_training_data(mode=sys.argv[1])
	
		self.src_max_len = 50
		self.tgt_max_len = 13
		self.src_token_size = len(self.src_ctoi)
		self.tgt_token_size = len(self.tgt_ctoi)
		print('token_size =', self.src_token_size, self.tgt_token_size)
		
		self.encoder_input = src_padded
		self.decoder_input = tgt_padded
		self.decoder_output = set_decoder_output_data(tgt_padded)
		#self.decoder_output = categorical_3d(self.decoder_output, self.tgt_max_len, self.tgt_token_size)

		self.cut_validation()
		self.target = np.zeros([self.batch_size, self.decoder_output.shape[1], self.tgt_token_size])
		
	def cut_validation(self):
		from sklearn.model_selection import train_test_split
		self.encoder_input, self.encoder_input_valid, self.decoder_input, self.decoder_input_valid, self.decoder_output, self.decoder_output_valid = train_test_split(
													self.encoder_input, self.decoder_input, self.decoder_output, test_size=0.1, random_state=42)

	def generate_batch_data(self, encoder_input, decoder_input, decoder_output):
		bs = self.batch_size
		data_num = decoder_output.shape[0]
		loopcount = data_num // bs
		
		while True:
			i = randint(0, loopcount-1)
			batch_index = [j for j in range(i*bs, (i+1)*bs)]
			yield [encoder_input[batch_index], decoder_input[batch_index]], decoder_output[batch_index]
			gc.collect()

	def train(self): # train seq2seq and seq2seq_atten
		print('\n\nStart training')
		
		from keras.models import load_model
		import keras.losses
		keras.losses.custom_loss = masked_perplexity_loss
		model_filename = 'epoch13'
		self.seq2seq_model = load_model('saved_model/seq2seq_'+model_filename+'.h5', custom_objects={'masked_perplexity_loss': masked_perplexity_loss})
		self.encoder_model = load_model('saved_model/encoder_'+model_filename+'.h5', custom_objects={'masked_perplexity_loss': masked_perplexity_loss})
		self.decoder_model = load_model('saved_model/decoder_'+model_filename+'.h5', custom_objects={'masked_perplexity_loss': masked_perplexity_loss})
		print('load model finished\n\n')
		
		# model.save_weights('my_model_weights.h5')
		# model.load_weights('my_model_weights.h5')
		#self.seq2seq_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', loss_weights=self.loss_weights)
		#self.decoder_target_data = to_categorical(self.decoder_target_data, num_classes = self.token_output_size)

		
		from keras import optimizers
		#self.seq2seq_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
		#adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, clipvalue=5.0) # amsgrad=False, 
		self.seq2seq_model.compile(optimizer='adam', loss=masked_perplexity_loss)
		from math import ceil
		steps_per_epoch = ceil(self.decoder_output.shape[0] / self.batch_size)
		validation_steps = ceil(self.decoder_output_valid.shape[0] / self.batch_size)
		print('loss = masked_perplexity_loss')
		for epoch in range(1, self.epoch+1):
			#self.seq2seq_model.fit([self.encoder_input, self.decoder_input], self.decoder_output, batch_size=512, epochs=1, validation_split=0.1, shuffle = False, verbose = 1)
			self.seq2seq_model.fit_generator(self.generate_batch_data(self.encoder_input, self.decoder_input, self.decoder_output),
				steps_per_epoch = steps_per_epoch,
				validation_data = self.generate_batch_data(self.encoder_input_valid, self.decoder_input_valid, self.decoder_output_valid),
				validation_steps = validation_steps) # default epochs = 1 in keras

			model_filename = 'epoch'+str(epoch)
			self.output_file = 'result/result_0312_'+model_filename
			self.test()
			if epoch % 5 == 0:	
				self.seq2seq_model.save('saved_model/seq2seq_'+model_filename+'.h5') 
				self.encoder_model.save('saved_model/encoder_'+model_filename+'.h5') 
				self.decoder_model.save('saved_model/decoder_'+model_filename+'.h5') 
				

	def inference_batch(self, input_seq, output=False):
		states_value = self.encoder_model.predict(input_seq)[0]  # Encode the input as state vectors.
		target_seq = np.zeros((input_seq.shape[0], 1)) # Generate empty target sequence of length 1.
		target_seq[:, 0] = np.tile(self.tgt_ctoi['SOS'], (input_seq.shape[0]))

		# Sampling loop for a batch of sequences (to simplify, here we assume a batch of size 1).
		stop_condition = np.zeros([input_seq.shape[0]])
		decoded_sentence = []
		for i in range(target_seq.shape[0]):
			decoded_sentence += [[]]
		while np.sum(stop_condition) < input_seq.shape[0]:
			output_tokens, states_value = self.decoder_model.predict([target_seq] + [states_value], verbose = 0)
			sampled_token_index = np.argmax(output_tokens[:, -1, :], axis = -1) # Sample a token
			for i in range(target_seq.shape[0]):
				if (sampled_token_index[i] == self.tgt_ctoi['EOS'] or len(decoded_sentence[i]) > self.tgt_max_len):
					#or sampled_token_index[i] <= self.tgt_ctoi['EOS']): # find stop or hit max --> OOV
					stop_condition[i] = 1
				if stop_condition[i] == 0: 
					decoded_sentence[i] += [self.tgt_itoc[sampled_token_index[i]]]  # (+= sampled_char)
					target_seq[i, 0] = sampled_token_index[i]
			if int(np.sum(stop_condition)) == input_seq.shape[0]: break
		for i in range(target_seq.shape[0]):
			if len(decoded_sentence[i]) == 0: decoded_sentence[i] += '我'
		
		return decoded_sentence

	def test(self, bs = 500):
		# note: 70000 % bs (batch_size) should be 0 
		results = []
		for i in range(int(len(self.testing_data)/bs)):
			if i % 10 == 0: print((i*bs))
			results += self.inference_batch(self.testing_data[i*bs:(i+1)*bs])
		utils.write_results(results, output_file = self.output_file)

	def load_model_check_acc(self):
		import utils
		from keras.models import load_model
		import keras.losses
		import pickle
		self.encoder_model,self.decoder_model = build_model.enc_dec(self.src_max_len, self.tgt_max_len, self.src_token_size, self.tgt_token_size)
		keras.losses.custom_loss = utils.masked_perplexity_loss
		model_filename = 'epoch10'
		model_path = '../src_seq2seq_0315/saved_model/' #len_rhy/'
		self.encoder_model.load_weights(model_path+'encoder_'+model_filename+'.h5')
		self.decoder_model.load_weights(model_path+'decoder_'+model_filename+'.h5')
		print('load model finished\n')

		#self.seq2seq_model, self.encoder_model, self.decoder_model = utils.load_model(model_filename = 'epoch10', model_path = 'saved_model/len_rhy')
		self.output_file = 'result/result_0503_epoch10'
		self.test()

		from gen_predict_features import create_features
		create_features(inputs=self.output_file)
		from check_acc import check_acc
		check_acc()

def main():
	mode = 'test'
	#teacher_forcing_mode = 

	lg = LyricGenerator()
	lg.load_data()

	lg.testing_data = read_data.load_testing_data(sys.argv[1])
	if mode == 'train':
		lg.seq2seq_model, lg.encoder_model, lg.decoder_model = build_model.seq2seq_fixed(
								lg.src_max_len, lg.tgt_max_len, lg.src_token_size, lg.tgt_token_size)
		lg.train()

	elif mode == 'test':
		lg.seq2seq_model, lg.encoder_model, lg.decoder_model = build_model.seq2seq(
								lg.src_max_len, lg.tgt_max_len, lg.src_token_size, lg.tgt_token_size)
		lg.load_model_check_acc()
		exit()
		import utils
		from keras.models import load_model
		import keras.losses
		import pickle

		lg.encoder_model,lg.decoder_model = build_model.enc_dec(lg.src_max_len, lg.tgt_max_len, lg.src_token_size, lg.tgt_token_size)
		keras.losses.custom_loss = utils.masked_perplexity_loss
		model_filename = 'epoch10'
		#model_path = '../src_training_0402/saved_model/len_rhy/'
		#model_path = 'saved_model/len_rhy/'
		model_path = 'saved_model/'
		try:
			os.mkdir('saved_feature_0501/'+model_filename+'/')
		except FileExistsError:
			pass
		lg.decoder_model.summary()

		lg.encoder_model.load_weights(model_path+'encoder_'+model_filename+'.h5')
		lg.decoder_model.load_weights(model_path+'decoder_'+model_filename+'.h5')
		print('load model finished-\n\n')

		src = lg.testing_data

		# for length or rhyme
		'''samples_idx, samples_dict = utils.select_samples(src, mode = 'length', max_len = 11)
		dense_weight = utils.get_dense_weight(lg.decoder_model) # [weight, bias] = [(128, 50006), (50006,)]
		dense_values = utils.get_dense_values(lg.encoder_model, lg.decoder_model, src, samples_idx, dense_weight[0].shape[1])
		
		utils.count_common_len(src)
		for selected_len in [4, 5, 6]:
			samples_idx, samples_dict = utils.select_samples(src, mode = 'rhyme', selected_len = selected_len)
			print('samples_idx =', samples_idx.shape, samples_idx[0, :10])
			
			filehandler = open('saved_feature_0501/'+model_filename+'/samples_rhyme_len='+str(selected_len)+'.pkl', 'wb')
			pickle.dump(samples_idx, filehandler)
			#samples_idx, samples_dict, samples_len = utils.select_samples(src, mode = 'rhyme', selected_len = selected_len, all_len = True)
			
			gate_values = utils.get_gate_values(lg.encoder_model, lg.decoder_model, src, samples_idx, pos_dict = None) # don't need to set mode if mode = length or rhyme
			filehandler = open('saved_feature_0501/'+model_filename+'/rhyme_len='+str(selected_len)+'.pkl', 'wb')
			#filehandler = open('saved_feature/rhyme_len=all.pkl', 'wb')
			pickle.dump(gate_values, filehandler)

			encoder_state = utils.get_encoder_state(lg.encoder_model, src, samples_idx)
			filehandler = open('saved_feature_0501/'+model_filename+'/encoder_rhyme_len='+str(selected_len)+'.pkl', 'wb')
			pickle.dump(encoder_state, filehandler)
			
		exit()
		dense_weight = utils.get_dense_weight(lg.decoder_model) # [weight, bias] = [(128, 50006), (50006,)]
		dense_values = utils.get_dense_values(lg.encoder_model, lg.decoder_model, src, samples_idx, dense_weight[0].shape[1])
		exit()
		
		
		#utils.check_weights(model)
		#utils.check_intermediate(model, src, tgt)
		'''
		# for pos
		#common_pos, pos_dict = utils.count_common_pos(src)
		pos_dict = {50071: 0, 50046: 1, 50057: 2}
		#print('pos_dict =', pos_dict)
		#exit()
		#import pdb
		#pdb.set_trace()
		samples_idx, samples_dict = utils.select_samples(src, mode = 'pos')	
		'''samples_idx, samples_dict = utils.select_samples(src, mode = 'length', max_len = 6, num = 10000)
		samples_idx = np.array(samples_idx)
		print('samples_idx =', samples_idx.shape) #len(samples_idx[0]))
		samples_idx = samples_idx[3:4]
		new_tmp = np.empty([1, 10000], dtype=int)
		for j in range(10000):
			new_tmp[0, j] = samples_idx[0][j]
		
		samples_idx = np.copy(new_tmp)'''

		print('samples_idx =', samples_idx.shape) #len(samples_idx[0]))
		gate_values = utils.get_gate_values(lg.encoder_model, lg.decoder_model, src, samples_idx, pos_dict = pos_dict, mode = 'pos') 
		#pdb.set_trace()
		filehandler = open('saved_feature_0503_lrp/pos_3classes.pkl', 'wb')
		pickle.dump(gate_values, filehandler)
		

if __name__ == '__main__':
	main()

