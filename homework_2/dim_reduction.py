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

def graph_data3D(data, fig_name):

	fig = plt.figure()

	ax = fig.add_subplot(111, projection='3d')

	color_list = ['r', 'g', 'b', 'y']

	for c in range(4):
		b = data[:,-1] == c
		x0 = data[:, 0][b]
		x1 = data[:, 1][b]
		x2 = data[:, 2][b]

		ax.scatter(x0, x1, x2, color=color_list[c])

	plt.savefig(fig_name)


def graph_data2D(data, groups, fig_name):

	color_list = ['r', 'g', 'b', 'y']

	for c in range(4):
		b = groups == c
		x0 = data[:, 0][b]
		x1 = data[:, 1][b]

		plt.scatter(x0, x1, color=color_list[c])

	plt.xlim((data[:,0].min(), data[:,0].max()))
	plt.ylim((data[:,1].min(), data[:,1].max()))
	plt.savefig(fig_name)


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


def do_LLE(data, out_dim, k):
	'''
	Does local linear embedding with k nearest neighbors
	Each ROW in data is an observation
	'''

	# Loop over each observation in the data
	N = data.shape[0]

	w_matrix = np.zeros( (N, N) )

	w_dict = {}

	for i in range(N):

		# Calculate the weights for that observation i
		cur_obs = data[i,:]
		local_data = data - cur_obs 
		local_distances = linalg.norm(local_data, axis=1)
		assert local_distances.shape == (N,)

		ordering = np.argsort(local_distances)
		assert ordering[0] == i 
		k_nn_indices = ordering[1:k+1]

		neighborhood = local_data[k_nn_indices]

		# create the local Gram matrix
		K_i = neighborhood @ neighborhood.T
		ones = np.ones((k,1))

		# Solve for w_i matrix and normalize
		w_i = linalg.inv(K_i) @ ones
		w_i = w_i / linalg.norm(w_i, axis=0)

		# we need to input the KNN w_i avlues into w_matrix
		#print("{} NN of pt {} are:".format(k, cur_obs))
		for j in range(k):
			neighbor_index = k_nn_indices[j]
			neighbor_weight = w_i[j]
			w_matrix[i,neighbor_index] = neighbor_weight
			#print("\t pt {}".format(data[neighbor_index,:]))

	# Construct matrix M from the sparse w_matrix
	n_I = np.identity(N)
	M = (n_I - w_matrix).T @ (n_I - w_matrix)

	e_vals, e_vecs = linalg.eigh(M)
	proj = e_vecs[:, 1:1+out_dim]

	return proj


def do_iso(data, k, proj_dim):

	N = data.shape[0]

	# step1: calcualte the pairwise distances
	pairwise_distances = np.empty( (N,N) )
	for i in range(N):
		for j in range(N):
			x_i = data[i,:]
			x_j = data[j,:]
			diff = x_i - x_j
			norm_diff = linalg.norm(diff)
			pairwise_distances[i,j] = norm_diff

	# step2: calculate the nearest neighbors for each column
	dist_w_inf = np.full( (N,N), np.inf)

	for i in range(N):
		dist_i = pairwise_distances[:,i]
		ordered_indices = np.argsort(dist_i)

		knn_indices = ordered_indices[0:k+1]
		dist_w_inf[knn_indices, i] = pairwise_distances[knn_indices, i]

	# step3: compute shortest path
	# Apply Floyd-Warshall algorithm
	short_path_dist = dist_w_inf
	for k in range(N):
		for i in range(N):
			for j in range(N):

				if short_path_dist[i,j] > short_path_dist[i,k] + short_path_dist[k,j]:
					short_path_dist[i,j] = short_path_dist[i,k] + short_path_dist[k,j]

	# Do eigen decomp on shortest paths
	e_vals, e_vecs = linalg.eigh(short_path_dist)
	flipped_e_vals = np.flip(e_vals, axis=0)
	flipped_e_vecs = np.flip(e_vecs, axis=1)

	# step4: now that the eigens are sorted largest>smallest take first p, take sq.root
	flipped_e_vals[proj_dim:] = 0
	lambda_sqrt = np.sqrt(flipped_e_vals)

	# step5: construct diagonal matrix
	diag_lambda = np.diagflat(lambda_sqrt)
	y_matrix = (flipped_e_vecs @ diag_lambda).T 

	return pairwise_distances, dist_w_inf, short_path_dist, y_matrix 

if __name__ == "__main__":

	# Load data
	orig_data = np.loadtxt('data/3Ddata.txt')
	orig_data[:, 3] = orig_data[:, 3] - 1
	data_3d = orig_data[:,0:3]
	centered_data = center_data(data_3d)

	var_cov = centered_data.T @ centered_data * (1/centered_data.shape[0]) 
	
	# A: Do pca
	pca_coeff = do_pca(data_3d, 2)
	graph_data3D(orig_data, "3d_data.png")
	plt.clf()
	graph_data2D(pca_coeff, orig_data[:,-1], "pca.png")
	plt.clf()

	# B: do ISOMAP
	pairwise_distances, dist_w_inf, short_path_dist, y_matrix  = do_iso(data_3d, 10, 2)
	y0 = y_matrix[0,:]
	y1 = y_matrix[1,:]
	group = orig_data[:,-1]


	color_list = ['r', 'g', 'b', 'y']

	for c in range(4):
		b = group == c

		plt.scatter(y0[b], y1[b], color=color_list[c])
	plt.savefig('isomap.png')
	plt.clf()

	# C: do LLE
	lle_proj = do_LLE(data_3d, 2, 10)
	graph_data2D(lle_proj, orig_data[:,-1], "lle.png")
	plt.clf()




