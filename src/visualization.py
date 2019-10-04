"""Visualize the hidden state of the trained Seq2Seq model by plotting line chart 
and dotted chart (after dimension reduction).
"""

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np 
import pdb

from sklearn.decomposition import PCA

matplotlib.use('Agg')


def plot_xy(hidden_state, title, name="Plot"):
	"""Plot gate values on line chart."""

	# TODO: Support other layers. 
	# TODO: Current version can only support decoder time step. (Time step counts from 0.)
	# One class has one figure.
	# Sample input hidden_state shape: (10, 12, 100, 16)
	
	subfigure_number = math.ceil(hidden_state.shape[-1] / 10)
	hidden_state = np.mean(hidden_state, axis=2)  # Average on each sample.
	print("(Plot) check shape =", hidden_state.shape)
	time_list = range(hidden_state.shape[1])  # TODO: Only support decoder.

	for class_index in range(hidden_state.shape[0]):  # TODO: for loop over each class.
		# row_index = int(class_index / col_num)
		# col_index = class_index % col_num

		x = hidden_state[class_index]
		print("(Plot) x.dim =", x.shape)

		fig, axs = plt.subplots(subfigure_number)  
		for i in range(subfigure_number+1):
			dim_list = range(i * 10, min((i+1) * 10, hidden_state.shape[-1]))
			for j in range(hidden_state.shape[0]): 
				for dim in dim_list:
					print("(Plot) dim =", dim, x[:, dim].shape)
					if subfigure_number == 1:
						axs.plot(time_list, x[:, dim], 'C' + str(dim))
					else:
						axs[i].plot(time_list, x[:, dim], 'C' + str(dim%10)) # label="dim = %d" % dim)
		
		# plt.xlabel("t")
		# plt.ylabel("Gate values")
		# plt.axis((0, hidden_state.shape[1], 0, 1))
		plt.title("hidden value of %s, %s" %(name, title[class_index]))
		plt.legend(["dim = " + str(dim) for dim in range(hidden_state.shape[-1])])
		plt.savefig('../tmp/Figure_plot_%s_%s' % (name, title[class_index]))
		plt.show()
		plt.close()
	

def scatter_with_dim_reduction(hidden_state, title, subtitle, reduction_method="PCA", name='Scatter'):
	"""Do dimension reduction and plot dotted chart."""
	
	# TODO: Use whole time steps as time_list.
	# Plot each time step in one figure.
	class_num = len(title)
	if class_num == 1:
		fig, axs = plt.subplots()
	else: 
		row_num = math.ceil(class_num ** 0.5)
		col_num = math.ceil(class_num / row_num)
		fig, axs = plt.subplots(row_num, col_num, figsize=(12, 8)) 
		print("(Scatter) row_num, col_num =", row_num, col_num) 

	if reduction_method == "PCA":
		for class_index in range(class_num):
			pca = PCA(n_components=2)
			hidden_state_transpose = hidden_state[class_index, :, :, :].reshape([-1, hidden_state.shape[-1]])
			embedding = pca.fit_transform(hidden_state_transpose)
			embedding = embedding.reshape([hidden_state.shape[1], hidden_state.shape[2], 2])  # Change shape to: (Class, sample, 2).
			pdb.set_trace()
			if class_num == 1:
				for i, label in enumerate(subtitle):
					axs.scatter(embedding[i, :, 0], embedding[i, :, 1], label=subtitle[i], s=10)
			else:
				row_index = int(class_index / col_num)
				col_index = class_index % col_num
				print("(Scatter) row, col =", row_index, col_index)
				for i, label in enumerate(subtitle):
					axs[row_index, col_index].scatter(embedding[i, :, 0], embedding[i, :, 1], label=subtitle[i], s=10)
				axs[row_index, col_index].set_title(title[class_index])
			# axs[row_num-1, col_num-1].legend(labels=subtitle, loc='best')

		fig.suptitle("hidden value of %s" %(name))  # Set a global title for each subfigure.
		plt.tight_layout()
		plt.legend() 
		plt.savefig("../tmp/Figure_scatter_%s" % name)  # TODO: Chnage figure saved path.
		plt.show()
		plt.close()

	else:
		# TODO: Support another dimension reduction method: t-SNE.
		print("Error: The dim reduction method %s is not supported." % reduction_method)


def plot_gate_values():
	# TODO: Call plot_xy.
	return 


def scatter_gate_values(hidden_state, reduction_method="PCA", name='test'):
	"""Call scatter_with_dim_reduction."""
	# TODO: Can only support task, control length, now.
	# hidden_state = (Class, time, sample, dim).
	
	
	"""
	# The first dim is plotted as different subfigures, the second is shown as different class in one subfigure.
	time_list = None
	if not time_list:
		time_list = range(hidden_state.shape[1])

	x = hidden_state.transpose([1, 0, 2, 3])
	title = ['time = ' + str(i) for i in range(1, 11)]
	x = x[3:7]
	title = ['time = ' + str(i) for i in range(3, 7)]
	print('name =', name)
	subtitle = ['Length = ' + str(i) for i in range(1, 11)]
	scatter_with_dim_reduction(x, title, subtitle, reduction_method, name="time_"+name)
	
	# Plot each class in different subfigures.
	title = ['Length = ' + str(i) for i in range(1, 10)]
	subtitle = ['time = ' + str(i) for i in range(1, 10)]
	hidden_state = hidden_state[:, :10, :, :]
	# TODO: There is an error in showing legend. (subtitle)
	scatter_with_dim_reduction(hidden_state, title, subtitle, reduction_method, name="class_"+name)
	"""

	# Plot the same rt as the same label.
	# scatter_with_dim_reduction(x, sample)
	# ((10-10, 9-9, 8-8), (10-9, 9-8, 8-7), (10-8, 9-7, 8-6), (10-7, 9-6, 8-5), (10-6, 9-5, 8-4)) (class-time)
	# 5 different rt, each with 3 time step
	num_rt = 5
	num_each = 6
	x = np.empty([1, num_rt, num_each*hidden_state.shape[2], hidden_state.shape[3]])
	N = hidden_state.shape[2]
	for i in range(0, num_rt):
		for j, jj in enumerate(range(10, 10-num_each, -1)):
			x[0, i, j*N:(j+1)*N] = hidden_state[jj-1, jj-i-1]
	title = ['rt']
	subtitle = ['rt = ' + str(i) for i in range(0, num_rt)]
	scatter_with_dim_reduction(x, title, subtitle, reduction_method, name="rt,rt=%d,each=%d_%s" % (num_rt, num_each, name))

