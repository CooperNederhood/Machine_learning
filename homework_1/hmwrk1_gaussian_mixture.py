import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def calc_normal_matrix(data, k, mean_vec, cov_mat):
	'''
	Given data, integer k denoting number of mixtures, and
	the corresponding means and covariance, returns the matrix of 
	multiv normal probabilities

	Inputs:
		- data: (np array - N x d) of input data
		- k: (int) number of Gaussian mixtures
		- mean_vec: (np array - k x d) of Gaussian centers
		- cov_mat: (np array - k x d x d) of Gaussian var-covar matrices

	Returns:
		- pdf_matrix: (np array - N x k) of multiv normal pdf for data
	'''

	N = data.shape[0]

	pdf_matrix = np.empty( (N, k) )

	for cur_cluster in range(k):
		cluster_pdf = multivariate_normal.pdf(data, mean=mean_vec[cur_cluster], cov=cov_mat[cur_cluster])
		pdf_matrix[:, cur_cluster] = cluster_pdf

	return pdf_matrix 

def calc_cluster_prob(data, k, mean_vec, cov_mat, pi_vec):
	'''
	Given data, k mixtures with mean and covariance, and mixtures with
	priors of pi_vec, calculates the probability that each observation
	is in each respective k mixture 

	Inputs:
		- data: (np array - N x d) of input data
		- k: (int) number of Gaussian mixtures
		- mean_vec: (np array - k x d) of Gaussian centers
		- cov_mat: (np array - k x d x d) of Gaussian var-covar matrices
		- pi_vec: (np array - k x 1) prior probability of each mixture 

	Returns:
		- prob_mat: (np array - N x k) prob obs i is in cluster j
	'''
	N = data.shape[0]
	prob_mat = np.empty( (N, k) )

	normal_pdfs = calc_normal_matrix(data, k, mean_vec, cov_mat)

	prob_mat = normal_pdfs * pi_vec.T 
	prob_mat = prob_mat / prob_mat.sum(axis=1).reshape( (N,1) )

	return prob_mat 

def calc_expected_likelihood(data, k, mean_vec, cov_mat, pi_vec):
	'''
	Calculates the expected log likelihood given parameters. Needed
	to test convergence of Gaussian Mixture algorithm

	Inputs:
		- data: (np array - N x d) of input data
		- k: (int) number of Gaussian mixtures
		- mean_vec: (np array - k x d) of Gaussian centers
		- cov_mat: (np array - k x d x d) of Gaussian var-covar matrices
		- pi_vec: (np array - k x 1) prior probability of each mixture 

	Returns:
		- exp_ll: (float) expected log likelihood
	'''

	log_like_ij = np.log(pi_vec).T + np.log(calc_normal_matrix(data, k, mean_vec, cov_mat))
	prob_in_cluster = calc_cluster_prob(data, k, mean_vec, cov_mat, pi_vec)

	exp_ll_matrix = log_like_ij * prob_in_cluster

	exp_ll = np.sum(exp_ll_matrix)
	return exp_ll

def calc_m_step(data, k, prob_mat):
	'''
	Given data, integer k, and updated probability assignments,
	update the pi_vec, mean_vec, and cov_mat

	Inputs:
		- data: (np array - N x d) of input data
		- k: (int) number of Gaussian mixtures
		- prob_mat: (np array - N x k) prob obs i is in cluster j

	Returns: a tuple containing (in order)... 
		- mean_vec: (np array - k x d) of Gaussian centers
		- cov_mat: (np array - k x d x d) of Gaussian var-covar matrices
		- pi_vec: (np array - k x 1) prior probability of each mixture 
	'''
	N = data.shape[0]
	d = data.shape[1]

	pi_vec = prob_mat.mean(axis=0)

	weights = prob_mat / prob_mat.sum(axis=0).reshape( (1,3) )
	assert np.abs(weights.sum() - k) < 0.01

	mean_vec = np.empty( (k,d) )
	for cluster_num in range(k):
		product = data * (weights[:, cluster_num].reshape( (N,1) ))
		center = product.sum(axis = 0)
		mean_vec[cluster_num] = center

	cov_mat = np.empty( (k, d, d) )

	for cluster_num in range(k):
		covar = np.zeros( (d,d) )
		centered_mean = data - mean_vec[cluster_num]
		for i in range(N):
			obs = centered_mean[i].reshape( (1,d) )
			covar += (obs.T @ obs) * weights[i,cluster_num]
		cov_mat[cluster_num] = covar 

	return mean_vec, cov_mat, pi_vec

def initialize_k_mixture(data, k):
	'''
	Given data and k desired mixtures, returns initial 
	values for the target parameters

	Inputs:
		- data: (np array - N x d) of input data
		- k: (int) number of Gaussian mixtures

	Returns: a tuple containing (in order)...
		- mean_vec: (np array - k x d) of Gaussian centers
		- cov_mat: (np array - k x d x d) of Gaussian var-covar matrices
		- pi_vec: (np array - k x 1) prior probability of each mixture 
	'''
	N = data.shape[0]

	rand_indeces = np.random.randint(0, N, size=k)
	mean_vec = data[rand_indeces,:]

	pi_vec = np.full( (k,1), 1/k )

	centered = data - data.mean(axis=0)
	cov_mat = np.array( [(centered.T @ centered) / N]*k ) 

	return mean_vec, cov_mat, pi_vec  


def k_gaussian_mixture(data, k, conv_tolerance, mean_vec=None, cov_mat=None, pi_vec=None, conv_ts=None):
	'''
	Given data and desired k gaussian mixtures, finds MLE for parameters

	Inputs:
		- data: (np array - N x d) of input data
		- k: (int) number of Gaussian mixtures
		- conv_tolerance: (float) algorithm converges once exp LL changes by less than this amount
		- mean_vec: (np array - k x d) of Gaussian centers
		- cov_mat: (np array - k x d x d) of Gaussian var-covar matrices
		- pi_vec: (np array - k x 1) prior probability of each mixture 

	Output:
		- TBD
	'''

	if mean_vec is None:
		mean_vec, cov_mat, pi_vec = initialize_k_mixture(data, k)
		conv_ts = []

	cur_exp_ll = calc_expected_likelihood(data, k, mean_vec, cov_mat, pi_vec)
	conv_ts.append(cur_exp_ll)

	prob_mat = calc_cluster_prob(data, k, mean_vec, cov_mat, pi_vec)
	new_mean_vec, new_cov_mat, new_pi_vec = calc_m_step(data, k, prob_mat)

	new_exp_ll = calc_expected_likelihood(data, k, new_mean_vec, new_cov_mat, new_pi_vec)

	if np.abs(cur_exp_ll - new_exp_ll) < conv_tolerance:
		return (new_mean_vec, new_cov_mat, new_pi_vec, conv_ts)

	else:
		return k_gaussian_mixture(data, k, conv_tolerance, new_mean_vec, new_cov_mat, new_pi_vec, conv_ts)

def plot_dist_graph(data, k, iterations, conv, file_name):
	'''
	Creates plot of exp log likelihood graph through iteration
	'''

	for _ in range(iterations):

		dist_series = k_gaussian_mixture(data, k, conv)[3]
		plt.plot(dist_series, color='black', alpha=.7)

	plt.xlabel("Iteration")
	plt.ylabel("Exp log likelihood")
	title = "Toy data k-means Gaussian mixture"
	plt.title(title)
	plt.savefig(file_name+".png", format='png')

# 3.g - Create graph of 2D assignment and compare convergence to k-means

toy_data = np.loadtxt('toydata.txt')
N = toy_data.shape[0]
k = 3
conv = 0.1

mean, cov, pi, ts = k_gaussian_mixture(toy_data, k, conv)

# 2D assignment graph, assign to largest p_(i,j)
prob_matrix = calc_cluster_prob(toy_data, k, mean, cov, pi)
assignments = np.argmin(prob_matrix, axis = 1).reshape( (N,1) )

plt.clf()
colors = ['red', 'orange', 'green']
for cluster in range(k):
	b = assignments == cluster 
	b.reshape(N)	
	plt.scatter(toy_data[b.reshape(N),0], toy_data[b.reshape(N),1], color=colors[cluster], alpha=.3)
plt.scatter(mean[:,0], mean[:,1], color='black')
plt.title('Toy data gaussian mixture k=3')
plt.savefig('2D_gaussian_mixture.png', format='png')

# Time series of convergence thru 20 runs 
plt.clf()
plot_dist_graph(toy_data, k, 20, conv, "distortion_gaussian")
