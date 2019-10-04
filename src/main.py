"""Main part of explainable Seq2Seq.
"""


import numpy as np
import os
import pdb
import pickle

import analysis
import analyze_weight
import build_model	
import visualization

from Seq2Seq import Seq2Seq 


def set_decoder_output_data(decoder_input):
	# TODO: Remove this function to other file.
	decoder_output = decoder_input.copy()
	for i in range(len(decoder_output)): 
		decoder_output[i, :-1] = decoder_input[i, 1:]  # remove the first token in decoder output
		decoder_output[i, -1] *= 0
	decoder_output = np.reshape(decoder_output, [decoder_output.shape[0], decoder_output.shape[1], 1])
	return decoder_output


def main():
	# TODO: Add a function to parse arguments.
	
	# Global setting here.
	mode_list = ["train", "visualization", "analysis"]
	mode = mode_list[0]
	task = "control_length"
	# Each model has its own file. User can add information (epochs, randon seed, task) in the file name.
	model_path = "../saved_model/control_length_units=4" 
	
	if mode == "train" or mode == "analysis":
		units = 4
	
	
	if mode == 'train':
		# Train a Seq2Seq model.
		# Define other variables only used in train mode.
		epochs = 10
		log_file = "training_control_length_units=4.txt"

		seq2seq = Seq2Seq(mode, task, model_path, log_file, units, epochs)
		seq2seq.seq2seq_model, seq2seq.encoder_model, seq2seq.decoder_model = build_model.seq2seq(
			seq2seq.src_max_len, 
			seq2seq.tgt_max_len, 
			seq2seq.src_token_size, 
			seq2seq.tgt_token_size, 
			latent_dim=seq2seq.units)
		seq2seq.train()
		seq2seq.check_accuracy(check_list=["word", "length"])  # Check accuracy after training.
		print("data =", seq2seq.data)
		print("units =", seq2seq.units)
		print("=" * 50)


	if mode == "analysis":
		# Evaluate a trained Seq2Seq and get trained weights and hidden values from the model.
		seq2seq = Seq2Seq(mode, task, model_path, units=units)
		seq2seq.seq2seq_model, seq2seq.encoder_model, seq2seq.decoder_model = build_model.seq2seq(
			seq2seq.src_max_len, 
			seq2seq.tgt_max_len, 
			seq2seq.src_token_size, 
			seq2seq.tgt_token_size, 
			latent_dim = seq2seq.units)
		seq2seq.load_seq2seq(model_path)
		# seq2seq.check_accuracy(check_list=["word", "length"])
		
		# Get the trained weights and save them.
		weights = analysis.get_gru_weights(seq2seq)
		with open(os.path.join(model_path, 'gate_values'), 'rb') as filehandler:
			gate_values = pickle.load(filehandler)
		analysis_weight.main(weights, gate_values)
		return

		# Get hidden state by their labels and save them.
		# TODO: Parse N (sample number of each label and each time step) into this function.
		sample = analysis.get_sample(task)
		
		# Hidden values in the last fully connected layer.
		dense_values = analysis.get_dense_values(seq2seq, sample)
		with open(os.path.join(model_path, 'dense_values'), 'wb') as filehandler:
			print('(Main) dense_values =', dense_values.shape)  # Shape = (10, 12, 100, 8)
			pickle.dump(dense_values, filehandler)

		# Hidden values in the decoder GRU layer.
		hidden_state = analysis.get_hidden_state(seq2seq, sample)  # Shape = (10, 100, 12, 16)
		hidden_state = hidden_state.transpose([0, 2, 1, 3])
		with open(os.path.join(model_path, 'hidden_state'), 'wb') as filehandler:
			print('(Main) hidden_state =', hidden_state.shape)  # Shape = (10, 12, 100, 16)
			pickle.dump(hidden_state, filehandler)

		# Gate values in the decoder GRU layer.
		gate_values = analysis.get_gate_values(seq2seq, sample)
		with open(os.path.join(model_path, 'gate_values'), 'wb') as filehandler:
			print('(Main) gate_values[z] =', gate_values['z'].shape)  # Shape = (10, 12, 100, 16)
			pickle.dump(gate_values, filehandler)

		# pdb.set_trace()
		# analysis.dim_reduction_plot(hidden_state, sample)

	
	if mode == 'visualization':
		# Load the hidden state and visualize it.
		# TODO: Move this part to visualization.main().
		# Load values.
		with open(os.path.join(model_path, 'gate_values'), 'rb') as filehandler:
			gate_values = pickle.load(filehandler)
		if task == "control_length":
			title = ['Length=' + str(i) for i in range(1, 11)]
			for gate in ['h', 'z', 'r', 'hh']:
				visualization.plot_xy(gate_values[gate], title, name='%s' % gate)
				# return
			
			for gate in ['h', 'z', 'r', 'hh']:
				visualization.scatter_gate_values(gate_values[gate], name=gate)
				break


if __name__ == "__main__":
	main()
	
