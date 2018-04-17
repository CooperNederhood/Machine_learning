import numpy as np 
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def center_data(raw_data):
	'''
	Given raw_data as numpy array, where a row
	is an observation. Return centered data

	Input: raw_data (numpy array)
	Returns: centered_data (numpy array)
	'''

	col_mean = raw_data.mean(axis=0)
	centered_data = raw_data - col_mean
	return centered_data

def graph_data3D(data):

	fig = plt.figure()

	ax = fig.add_subplot(111, projection='3d')

	color_list = ['r', 'g', 'b', 'y']

	for c in range(4):
		b = data[:,-1] == c
		x0 = data[:, 0][b]
		x1 = data[:, 1][b]
		x2 = data[:, 2][b]

		ax.scatter(x0, x1, x2, color=color_list[c])

	plt.show()

def graph_data2D(data, groups):

	color_list = ['r', 'g', 'b', 'y']

	for c in range(4):
		b = groups == c
		x0 = data[:, 0][b]
		x1 = data[:, 1][b]

		plt.scatter(x0, x1, color=color_list[c])

	plt.xlim((data[:,0].min(), data[:,0].max()))
	plt.ylim((data[:,1].min(), data[:,1].max()))
	plt.show()

def do_pca(data, comp_num):
	'''
	Return original data mapped 
	to first comp_num of princ components

	Inputs:
		- data: (numpy array) 
		- comp_num: (int) of prin comp
	Returns:
		- proj_data: (numpy array) of projected data
	'''

	centered_data = center_data(data)
	var_cov = centered_data.T @ centered_data * (1/centered_data.shape[0])

	# note: e_vecs will have COLUMNs as the corresponding eigenvectors
	# note: the e_vecs are already normalized, per documentation
	e_vals, e_vecs = linalg.eig(var_cov)

	e_ordering = np.argsort(e_vals)[::-1]
	e_basis = e_vecs[e_ordering][:, 0:comp_num]

	coeff = centered_data @ e_basis

	return coeff  


if __name__ == "__main__":

	orig_data = np.loadtxt('data/3Ddata.txt')
	orig_data[:, 3] = orig_data[:, 3] - 1
	data_3d = orig_data[:,0:3]
	centered_data = center_data(data_3d)

	var_cov = centered_data.T @ centered_data * (1/centered_data.shape[0]) 
	pca_coeff = do_pca(data_3d, 2)

	graph_data3D(orig_data)
	graph_data2D(pca_coeff, orig_data[:,-1])