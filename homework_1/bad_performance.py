import cooper_nederhood_homework1 as cdn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


'''NOTE: use the gen_bad_data to randomly generate complex data
	that will require the k-means++ initialization, then once we have
	created such a dataset, simply read that file in, rather than 
	constantly creating a new random dataset'''


def gen_bad_data(clusters, cluster_size, data_range, output_filename, x1_count, x2_count):
	'''
	Given a desired number of clusters and a range,
	generates a random and complex distribution of clusters which
	can induce poor performance in basic k-means

	Inputs:
		- clusters: (int) of cluster numbers 
		- cluster_size: (int) of obs generated per cluster 
		- data_range: (tuple) denoting data bound
		- output_filename: (string) filename to save data to
		- x1_count, x2_count: (int) how to partition the space evenly
		NOTE: data generated from (0,0) to data_range tuple

	Returns:
		- bad_data: (np array - cluster_size*cluster x 2) of generated data 
	'''

	x1_max = data_range[0]
	x2_max = data_range[1]

	N = clusters 
	x1 = np.linspace(0,x1_max,x1_count)
	x2 = np.linspace(0,x2_max,x2_count)
	centers = []
	for i in x1:
		for j in x2:
			centers.append( (i,j) )
	assert len(centers) == 20

	scales = np.random.uniform(0.1, 1 ,size=N)
	scales = np.full(N, 1) 
	clusters = []

	for i in range(N):
		clusters.append(np.random.normal(loc=centers[i], scale=scales[i], size=(cluster_size,2)))

	bad_data = np.concatenate(clusters)

	f = open(output_filename, 'w')
	for i in range(bad_data.shape[0]):

		d = str(bad_data[i,0]) + "  " + str(bad_data[i,1])
		f.write(d)
		f.write('\n')
	f.close()


	return bad_data 

plt.clf()

k = 20
x1_count = 5
x2_count = 4
size = 20
d_range = (100, 100) 

bad_data = gen_bad_data(k, size, d_range, 'bad_data.txt', x1_count,x2_count)
#bad_data = np.loadtxt('bad_data.txt')

plt.scatter(bad_data[:,0], bad_data[:,1])
plt.show()

# Iterate for k-means++
plus_plus_distortion = []

for _ in range(k):

	ts = cdn.return_distortion(bad_data, k, pp=True)
	plus_plus_distortion.append(ts)
avg = np.sum(plus_plus_distortion)/k

# Run with simple k-means and save out 2D graph
colors = sns.color_palette(None, k)
centroids, assignments, dist_series = cdn.k_means(bad_data, k)

N = bad_data.shape[0]

for cluster in range(k):
	b = assignments == cluster 
	plt.scatter(bad_data[b.reshape(N),0], bad_data[b.reshape(N),1], color=colors[cluster], alpha=.3)
plt.scatter(centroids[:,0], centroids[:,1], color='black')
plt.title('Vanilla k-means poor performance')
plt.savefig('Poor_performance_example.png', format='png')
print("Vanilla k-means distortion = {}".format(dist_series[-1]))
