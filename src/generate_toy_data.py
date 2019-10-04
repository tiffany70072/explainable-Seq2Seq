"""Generate toy data.

Default training sample = 40000, testing sample = 10000.
	PAD_token = 0.
	SOS_token = 1.
	EOS_token = 2.

Input and output formats: 
	Word index, not one-hot word embedding.
	Word embedding will be processed in trainer.py.
Input and output size = (N, max_length + 2)
	because each sentence starts with SOS and ends with EOS.
	Without NOR, NOE, NOX

"""

import numpy as np
import os

np.random.seed(0)


def task1(name, max_length=5, 
		PAD_token = 0, SOS_token = 1, EOS_token = 2, 
		N = 50000, test_ratio = 0.2, specific_tokens = 3):
	""" 
	inputs = [1, 3, 5, 5, "3"] # fix len = 5, class_num = 10
	outputs = [0, 0, 0, 1, 0] # fix len = 5, class_num = 2 (0, 1)
	   [[6 4 7 4 4]
		 [9 9 4 3 0]
		 [5 9 6 0 3]
		 [4 1 4 4 3]
		 [7 3 3 2 3]] 
		 [[0 0 0 0 1]
		 [1 0 0 0 0]
		 [0 0 0 1 0]
		 [0 0 0 1 0]
		 [0 0 0 1 0]]
	"""
	
	target = np.random.randint(max_length, size = (N))
	outputs = np.eye(max_length, dtype = int)[target]                  

	inputs = np.random.randint(10, size = (N, 5))
	inputs[:, -1] = target 
	print(inputs[:10], outputs[:10])

	try: 
		os.mkdir(name)
	except FileExistsError: 
		pass
	np.save(name+"/in.npy", inputs)
	np.save(name+"/out.npy", outputs)


def control_length(name, vocab_size=5, min_length=1, max_length=10, 
				PAD_token=0, SOS_token=1, EOS_token=2, 
				N=50000, test_ratio=0.2, specific_tokens=3):
	"""
	[[ 0.  0.  0.  0.  0.  1.  3.  3.  7.  5.  4.  2. 10.]
	 [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  2.  4.]
	 [ 0.  0.  0.  0.  0.  0.  0.  1.  4.  3.  4.  2.  6.]
	 [ 0.  0.  0.  0.  0.  0.  0.  1.  6.  3.  6.  2.  3.]
	 [ 0.  0.  0.  1.  5.  6.  3.  4.  6.  6.  6.  2.  5.]]
	[[1. 4. 4. 4. 3. 5. 7. 6. 6. 5. 7. 2.]
	 [1. 5. 3. 3. 7. 2. 0. 0. 0. 0. 0. 0.]
	 [1. 3. 7. 4. 7. 4. 5. 2. 0. 0. 0. 0.]
	 [1. 5. 3. 4. 2. 0. 0. 0. 0. 0. 0. 0.]
	 [1. 4. 4. 4. 6. 6. 2. 0. 0. 0. 0. 0.]]
 	"""

 	# Target length of each sample.
	input_length = np.random.randint(min_length, max_length+1, size=(N))  
	output_length = np.random.randint(min_length, max_length+1, size=(N))

	# Construct input data.
	inputs = np.zeros([N, max_length + 3], dtype=int)  # SOS, xxx, EOS, length control signal.
	for i in range(N):
		one_input = np.random.randint(specific_tokens, specific_tokens+vocab_size, size=(input_length[i]))
		inputs[i][-input_length[i]-3] = SOS_token
		inputs[i][-input_length[i]-2:-2] = one_input
		inputs[i][-2] = EOS_token
		inputs[i][-1] = output_length[i] + specific_tokens + vocab_size - min_length  # Control signal.
		# Control signal starts from 8 (3 + 5) to 18 (8 + 10).
	print(inputs[:5])
	
	# Construct output data.
	outputs = np.zeros([N, max_length + 2], dtype=int)  # SOS, xxx, EOS
	for i in range(N):
		one_output = np.random.randint(specific_tokens, specific_tokens+vocab_size, size=(output_length[i]))
		outputs[i][0] = SOS_token
		outputs[i][1:output_length[i]+1] = one_output
		outputs[i][output_length[i]+1] = EOS_token
	print(outputs[:5])   
	
	# Save data.
	try: 
		os.mkdir(name)
	except FileExistsError: 
		pass
	np.save(name+"/in.npy", inputs)
	np.save(name+"/out.npy", outputs)


def main():
	# task1("data/task1")
	control_length("data/control_length")


if __name__ == "__main__":
	main()