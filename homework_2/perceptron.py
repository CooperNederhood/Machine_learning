import numpy as np 
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def do_perceptron(ydata, xdata, threshold, init_weights = None, control = 0):

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


if __name__ == "__main__":

	# train_y = np.array([1, 1, 1, 1, -1, -1, -1, -1]).reshape((8,1))
	# train_x = np.array([[0,1], [0.1, 1.1], [0, 3], [-1, 1], 
	# 					[1,1], [1.1, 1.1], [1, 3], [1, 5]])

	# weights, y_hat, y_bool  = do_perceptron(train_y, train_x, 0.00001)

	train_x = np.loadtxt('data/train35.digits')
	const = np.ones((train_x.shape[0],1))

	train_x_constant = np.concatenate( (train_x, const), axis=1)
	train_y = np.loadtxt('data/train35.labels').reshape( (2000,1) )

	weights, y_hat, y_bool  = do_perceptron(train_y, train_x_constant, 0.00001)
	e, d = online_perceptron(train_y, train_x_constant)
	cum = e.cumsum()
	ts = cum / d
	plt.plot(ts)
	plt.show()