import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(point1, point2):
	'''
	Calculates the 2-norm distance between 2 points in space.
	Can be broadcasted across numpy arrays

	Inputs:
		- point1: (numpy array)
		- point2: (numpy array)

	Returns:
		- dist: (float) euclidian distance
	'''

	sq_dis = (point1 - point2)**2
	dist = np.sqrt(sq_dis.sum(axis=1))

	return dist 

def calc_distortion(data, centroids, assignments, k):
	'''
	Given some data, centroids, and integer k, returns
	the distortion of the cluster assignments calculated
	as the total squared distance 

	Inputs: 
		- data: (np array - N x d) of input data
		- centroids: (np array - k x d) of centroids in feature space
		- assignments: (np array - N x 1) of cluster assignments
		- k: (int) number of clusters

	Returns:
		- distortion: (np array - N x 1) of cluster assignments
	'''

	N = data.shape[0]
	total_distortion = 0

	for centroid_num in range(k):
		b = assignments == centroid_num 
		bool_filter = b.reshape(N)
		cur_cluster = data[bool_filter]

		sq_dis = euclidean_distance(cur_cluster, centroids[centroid_num])**2

		total_distortion += sq_dis.sum()

	return total_distortion

def calc_assignments(data, centroids, k):
	'''
	Given some data, centroids, and integer k, returns
	the distortion minimizing set of cluster assignments

	Inputs: 
		- data: (np array - N x d) of input data
		- centroids: (np array - k x d) of centroids in feature space
		- k: (int) number of clusters

	Returns:
		- assignments: (np array - N x 1) of cluster assignments
	'''

	N = data.shape[0]

	# calculate the N x k matrix of euclidian distances
	distances = np.empty( (N, k) )

	for centroid_num in range(k):
		dist = euclidean_distance(data, centroids[centroid_num])

		distances[:,centroid_num] = dist 

	# assign to the centroid that minimizes the euclidean distance
	assignments = np.argmin(distances, axis = 1).reshape( (N,1) )

	return assignments

def calc_centroid(data, assignments, k):
	'''
	Given some data, cluster assignments, and integer k, returns 
	the distortion minimizing set of k cluster centroids 

	Inputs:
		- data: (np array - N x d) of input data
		- assignments: (np array - N x 1) of cluster assignments
		- k: (int) number of clusters

	Returns:
		- centroids: (np array - k x d) of centroids in feature space
	'''	

	N = data.shape[0]
	d = data.shape[1]

	centroids = np.empty( (k,d) )

	for centroid_num in range(k):
		b = assignments == centroid_num 
		bool_filter = b.reshape(N)
		new_centroid = data[bool_filter].mean(axis = 0)
		centroids[centroid_num,:] = new_centroid

	return centroids


def k_means(data, k, cur_centroids = None, cur_assignments = None, dist_time_series = None):
	'''
	Given some data and an integer k, performs k-means clustering.
	Algorithm is implemented recursively

	Inputs: 
		- data: (np array - N x d) of input data
		- k: (int) number of clusters
		- assignments: (np array - N x 1) of cluster assignments (optional)
		- centroids: (np array - k x d) of centroids in feature space (optional)

	Returns:
		- (tuple) of final centroids and assignments and distortion thru iterations
	'''	

	N = data.shape[0]

	# Initialize random centroids if none are given
	if cur_centroids is None:
		rand_indeces = np.random.randint(0, N, size=k)
		cur_centroids = data[rand_indeces,:]

	if dist_time_series is None:
		dist_time_series = []

	new_assignments = calc_assignments(data, cur_centroids, k)

	if np.all(new_assignments == cur_assignments):
		return (cur_centroids, cur_assignments, dist_time_series)

	else:
		new_centroids = calc_centroid(data, new_assignments, k)
		dist_time_series.append(calc_distortion(data, new_centroids, new_assignments, k))
		return k_means(data, k, new_centroids, new_assignments, dist_time_series)

def plus_plus_init(data, k):
	'''
	Given data and an integer k, find the k-means++ optimal
	initial cluster centers pulled from the data

	Inputs: 
		- data: (np array - N x d) of input data
		- k: (int) number of clusters

	Returns:
		- centroids: (np array - k x d) of centroids in feature space
	'''

	N = data.shape[0]
	d = data.shape[1]
	centroids = np.empty( (k,d) )
	centroids[0,:] = data[ np.random.randint(0, N, size=1), :]

	for cur_k in range(1, k):

		distances = np.empty( (N, cur_k) )

		for centroid_num in range(cur_k):
			dist = euclidean_distance(data, centroids[centroid_num])

			distances[:,centroid_num] = dist 
		sq_min_distance = distances.min(axis=1)**2

		prob = sq_min_distance/sq_min_distance.sum()

		assert prob.shape == (N,)

		i = np.random.choice(range(0,N), size=1, p=prob)

		centroids[cur_k,:] = data[i, :]

	return centroids

def plot_dist_graph(data, k, iterations, file_name, plus_plus=False):
	'''
	Creates plot of distortion graph through iteration
	'''

	for _ in range(iterations):

		if plus_plus:
			init_centroids = plus_plus_init(toy_data, k)
		else:
			init_centroids = None

		dist_series = k_means(toy_data, k, cur_centroids = init_centroids)[2]
		plt.plot(dist_series, color='black', alpha=.7)

	plt.xlabel("Iteration")
	plt.ylabel("Distortion")
	init_type = "kmeans ++" if plus_plus else "random"
	title = "Toy data k-means distortion - {} initialization".format(init_type)
	plt.title(title)
	plt.savefig(file_name+".png", format='png')

def return_distortion(data, k, pp=False):
	'''
	Given some data and k, returns a distortion
	'''

	if pp:
		init_centroids = plus_plus_init(data, k)
	else:
		init_centroids = None 

	ts = k_means(data, k, cur_centroids = init_centroids)[2]
	return ts[-1]




if __name__ == "__main__":

	# Load data and set parameters
	toy_data = np.loadtxt('toydata.txt')
	N = toy_data.shape[0]
	k = 3

	centroids, assignments, dist_series = k_means(toy_data, k)

	# 2.a - 2D plot of assignments and plot of distortion function
	# 2D plot of assignments
	plt.clf()
	colors = ['red', 'orange', 'green']
	for cluster in range(k):
		b = assignments == cluster 
		b.reshape(N)	
		plt.scatter(toy_data[b.reshape(N),0], toy_data[b.reshape(N),1], color=colors[cluster], alpha=.3)
	plt.scatter(centroids[:,0], centroids[:,1], color='black')
	plt.title('Toy data k-means=3')
	plt.savefig('2D_kmeans.png', format='png')

	# Plot distortion function over 20 iterations
	plt.clf()
	plot_dist_graph(toy_data, k, 20, "distortion_kmeans")

	# 2.b - plot of distortion function using kmeans++ initialization
	plt.clf()
	plot_dist_graph(toy_data, k, 20, "distortion_kmeans_plus", plus_plus=True)




