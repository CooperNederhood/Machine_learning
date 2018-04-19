import numpy as np 
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def do_perceptron(ydata, xdata, threshold, init_weights = None, control = 0):
	'''
	NOTE: this is no longer used but i am keeping it along just
	to have a record of it and to work off of for the future. Does
	not do anything
	'''

	N = xdata.shape[0]
	d = xdata.shape[1]

	if init_weights is None:
		weights = np.zeros( (d,1) )
	else:
		weights = init_weights

	y_hat = xdata @ weights 
	y_hat[y_hat<0] = -1
	y_hat[y_hat>=0] = 1
	y_bool = (y_hat != ydata).astype(int)

	delta_weight = (xdata * y_bool * ydata).sum(axis=0).reshape( (d,1) )

	if control > 10:
		print("Over control sequence limit")
		return weights, y_hat, y_bool

	elif y_bool.sum() > 0:
		print("Making recursive call, errors = {}".format(y_bool.sum()))
		return do_perceptron(ydata, xdata, threshold, init_weights=(weights+delta_weight), control=control+1)

	else:
		print("DONE")
		return weights, y_hat, y_bool 

def online_perceptron(ydata, xdata):
	'''
	Performs online perceptron algorithm
	'''

	N = xdata.shape[0]
	d = xdata.shape[1]

	weights = np.zeros( (d,1) )

	error = np.zeros( N )
	denom = np.array(range(1,N+1))

	for i in range(N):
		cur_x = xdata[i].reshape( (1,d) )
		cur_y = ydata[i]
		# print(cur_x.shape)
		# print(weights.shape)
		# print('\n')
		y_hat = cur_x @ weights 
		# print(y_hat.shape)
		y_hat = -1 if y_hat < 0 else 1

		cur_y = int(cur_y)

		# print("y_hat={}".format(y_hat))
		# print("cur_y={}".format(cur_y))
		if y_hat != cur_y:
			error[i] = 1
			weights = weights + cur_y * cur_x.T 

	
	return error, denom 


def batch_perceptron(ydata, xdata, weights=None, c=0, error_hist=None):
	'''
	Performs batch perceptron algorithm
	'''

	N = xdata.shape[0]
	d = xdata.shape[1]

	if weights is None:
		weights = np.zeros( (d,1) )
		error_hist = []
	
	error = np.zeros( N )
	delta = np.zeros( (d,1) )

	for i in range(N):
		cur_x = xdata[i].reshape( (1,d) )
		cur_y = ydata[i]

		y_hat = cur_x @ weights 

		y_hat = -1 if y_hat < 0 else 1

		cur_y = int(cur_y)

		if y_hat != cur_y:
			error[i] = 1
			delta = delta + cur_y * cur_x.T 

	weights = weights + delta 
	#print("Run #{}, errors={}".format(c, error.sum()))

	if error.sum() == 0:
		return weights, error_hist
	else:
		return batch_perceptron(ydata, xdata, weights,c=c+1, error_hist = error_hist+[error.sum()])
	
def test_perceptron(xdata, weights, out_file):
	'''
	Given x-testing data and pre-trained weights,
	generates the y-predictions and outputs to file
	'''

	N = xdata.shape[0]
	d = xdata.shape[1]

	f = open(out_file, 'w')

	for i in range(N):
		cur_x = xdata[i].reshape( (1,d) )

		y_hat = cur_x @ weights 
		y_hat = -1 if y_hat < 0 else 1

		f.write(str(y_hat))
		f.write("\n")
	f.close()


if __name__ == "__main__":

	# Load data
	train_x = np.loadtxt('data/train35.digits')
	test_x = np.loadtxt('data/test35-1.digits')
	train_y = np.loadtxt('data/train35.labels').reshape( (2000,1) )

	# Create x constant column
	const = np.ones((train_x.shape[0],1))
	const_test = np.ones((test_x.shape[0],1))

	train_x_constant = np.concatenate( (train_x, const), axis=1)
	test_x_constant = np.concatenate( (test_x, const_test), axis=1)

	# Per suggestion on homework, normalize to 1
	train_x_unity = train_x / (linalg.norm(train_x, axis=1).reshape( (2000,1) ) )
	train_y_unity = train_y / (linalg.norm(train_y, axis=1).reshape( (2000,1) ) )
	
	# Run on raw data: no change
	weights_raw, ts_raw = batch_perceptron(train_y, train_x)
	print("Raw data: len = {}".format(len(ts_raw)))

	# Add bias term:
	weights_bias, ts_bias = batch_perceptron(train_y, train_x_constant)
	print("Add bias term data: len = {}".format(len(ts_bias)))

	# Normalize x-y data to norm 1
	weights_unit, ts_unit = batch_perceptron(train_y_unity, train_x_unity)
	print("Normalize data: len = {}".format(len(ts_unit)))

	# Don't see meaninglful difference between the three. Use run with bias term added
	x_label = list(range(1,len(ts_bias)+1))
	plt.clf()
	plt.plot(x_label, ts_bias)
	plt.title("Total misclassifications per training run")
	plt.xlabel('Iteration # through 2000pt training data')
	plt.ylabel('Total misclassifications in run')
	plt.savefig('Perceptron_training.png')

	test_perceptron(test_x_constant, weights_bias, "test35.predictions")
