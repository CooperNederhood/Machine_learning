import numpy as np 
import numpy.linalg as linalg
import skimage.color as skimage
import PIL 

D = 64
TRAINING_SIZE = 200
STRIDE = 1
SCALE_FILTERS = 1

class WeakLearner:
	"""A simple container class to store information defining a given 
		weak learner"""

	def __init__(self, i, theta, pol):
		self.i = i 
		self.theta = theta 
		self.polarity = pol 
		self.error = None
		self.alpha = None
		self.z = None 


	def __str__(self):

		str_1 = "Feature #{}:\n".format(self.i)
		str_2 = "\t theta={}\n".format(self.theta)
		str_3 = "\t poliarity={}\n".format(self.polarity)
		str_4 = "\t error={}\n".format(self.error)

		return str_1+str_2+str_3+str_4

	def calc_hypoth(self):
		'''
		Returns an array of the predicted y-values
		for the global dataset
		'''

		ft_vals = np.empty( TRAINING_SIZE*2 )

		for pic_num in range(TRAINING_SIZE*2):
			ft_vals[pic_num] = compute_feature(pic_num, self.i)

		
		y_hypoth = np.sign(self.polarity * (ft_vals - self.theta))

		return y_hypoth


def load_training(size):
	'''
	Given an integer of the size of the data in 
	each category, loads training data, and returns 
	II-lookup table

	Converts images to grayscale

	Inputs: size (int)
	Returns: ii_table (np array dim: size*2 x 64 x 64)
	'''

	ii_table = np.empty( (size*2, D, D) )

	# background images - coded as negatives
	for i in range(size):
		im = PIL.Image.open("background/{}.jpg".format(i))
		grey_im = skimage.rgb2grey( np.array(im) )
		assert grey_im.shape == (D, D)

		cum_sum = grey_im.cumsum(axis=0).cumsum(axis=1)

		ii_table[i] = cum_sum

	# face images - coded as positives
	for i in range(size):
		im = PIL.Image.open("faces/face{}.jpg".format(i))
		grey_im = skimage.rgb2grey( np.array(im) )
		assert grey_im.shape == (D, D)

		cum_sum = grey_im.cumsum(axis=0).cumsum(axis=1)

		ii_table[i+size] = cum_sum

	return ii_table 

def new_features(dimension, stride, primitive):
	'''
	Define the length=8 list of a primitive filter location.
	Given the dimensions of the testing image and the desired stride,
	return coordinates of all possible locations for the primitive

	Returns:
		filter_coords: array of all coord locations
	'''

	p0 = primitive[0:2]
	p1 = primitive[2:4]
	q0 = primitive[4:6]
	q1 = primitive[6:8]
	points = [p0, p1, q0, q1]

	p1_reset = p1[1]
	q1_reset = q1[1]

	filter_coords = []
	p1_ex = [i-1 for i in p1]
	q1_ex = [i-1 for i in q1]

	filter_coords.append( p0+p1_ex+q0+q1_ex )

	print("p0 initialized: {}".format(p0))
	print("p1 initialized: {}".format(p1_ex))
	print("q0 initialized: {}".format(q0))
	print("q1 initialized: {}".format(q1_ex))
	print("")

	while q1 != [dimension,dimension]:

		# try to go right
		if p1[1] != dimension:
			for pt in points:
				pt[1] += stride

		# if not, then we need to reset and move down
		else:
			p0[1] = 0
			p1[1] = p1_reset
			q0[1] = 0
			q1[1] = q1_reset

			for pt in points:
				pt[0] += stride

		p1_ex = [i-1 for i in p1]
		q1_ex = [i-1 for i in q1]

		print("p0 is: {}".format(p0))
		print("p1 is: {}".format(p1_ex))
		print("q0 is: {}".format(q0))
		print("q1 is: {}".format(q1_ex))
		print("")

		# Add points to the filter_coords list
		filter_coords.append( p0+p1_ex+q0+q1_ex )

	return np.array(filter_coords)


def return_II(II_table, top_left, bot_right, verbose=False):
	'''
	Given a summation table, and two coordinates corresponding
	to the top-left and bottom-right corners, returns the value of
	the specified region
	'''

	x0, y0 = top_left[0], top_left[1]
	x1, y1 = bot_right[0], bot_right[1]

	if verbose:
		print("Top left = ", top_left)
		print("Bottom right =", bot_right)
		print("Dimension of lookup table =", II_table.shape())


	lookup1 = II_table[x1, y1]
	lookup2 = 0 if y0 == 0 else II_table[x1, y0-1]
	lookup3 = 0 if x0 == 0 else II_table[x0-1, y1]
	lookup4 = 0 if x0 == 0 or y0 == 0 else II_table[x0-1, y0-1]

	return lookup1 - lookup2 - lookup3 + lookup4

def compute_feature(pic_number, ft_number):
	'''
	Computes the value of feature # ft_number on
	image # pic_number
	'''

	# Get the respective ii_table and feature coords from our globals
	ii_table = II_TABLES[pic_number]
	ft_coords = FEATURE_COORDS[ft_number]

	pos_weight = return_II(ii_table, ft_coords[0:2], ft_coords[2:4])
	neg_weight = return_II(ii_table, ft_coords[4:6], ft_coords[6:8])

	return pos_weight - neg_weight

def find_p_theta(ft_number, cur_weights):
	'''
	Before we can find the best weak learner, for each feature we must find
	the optimal cutoff theta and the polarity, p, which defines the best 
	weak learner for this featuer

	has_image denotes positive polarity

	Returns:
		- weak_learner: (WeakLearner type) containing ft_number, theta, pol, error
	'''

	ft_vals = np.empty( TRAINING_SIZE*2 )

	for pic_num in range(TRAINING_SIZE*2):
		ft_vals[pic_num] = compute_feature(pic_num, ft_number)

	ranking = np.argsort(ft_vals)

	sorted_weights = cur_weights[ranking]
	sorted_is_image = IS_IMAGE[ranking]
	sorted_is_background = IS_BACKGROUND[ranking]
	sorted_ft_vals = ft_vals[ranking]

	cum_image = np.cumsum(sorted_weights*sorted_is_image)
	cum_background = np.cumsum(sorted_weights*sorted_is_background)

	T_img = cum_image[-1]
	T_back = cum_background[-1]

	e = np.fmin(cum_image + (T_back - cum_background), cum_background + (T_img - cum_image) )

	min_e = np.argmin(e)
	minimum_error = e[min_e]

	#assert minimum_error > 0 and minimum_error <= .50
	assert minimum_error > 0

	if minimum_error > .5:
		print("Feature #{} has min error of {}".format(ft_number, minimum_error))

	theta = (sorted_ft_vals[min_e] + sorted_ft_vals[min_e+1] ) / 2
	polarity = 1 if cum_image[min_e] + (T_back - cum_background[min_e]) > cum_background[min_e] + (T_img - cum_image[min_e]) else -1

	weak_learner = WeakLearner(ft_number, theta, polarity)
	weak_learner.error = minimum_error

	return weak_learner


def best_learner(cur_weights):
	'''
	Given the current weights, returns the i-feature #, the polarity, and theta-cutoff
	which minimizes the error. i.e. this is our best weak classifier

	Returns:
		- i: number of the feature
		- polarity: +1/-1 of the feature
		- theta: cutoff value of the weak learner
	'''

	feature_count = len(FEATURE_COORDS)

	opt_learner = None
	min_error = np.inf 

	# Loop over all the features and get weak learner info
	for ft_num in range(feature_count):

		cur_learner = find_p_theta(ft_num, cur_weights)

		#print(cur_learner)
		#print("")

		if cur_learner.error < min_error:
			opt_learner = cur_learner
			min_error = cur_learner.error

			print("Feature #{} is NEW BEST CLASSIFIER".format(cur_learner.i))
			
	return opt_learner


def update_weights(cur_weights, bl):
	'''
	Given the current set of weights and the new weak
	learner to be added to the boosting hypothesis,
	returns the re-calculated weights 

	Inputs:
		- cur_weights: current weights of the training pop
		- bl: (WeakLearner) best weak learner in this boosting round

	Returns:
		- new_weights (np array), unweighted error of weak-learner
	'''

	y_true = IS_IMAGE - IS_BACKGROUND
	y_hypoth = bl.calc_hypoth()

	exponent = -bl.alpha * y_true * y_hypoth

	new_weights = (cur_weights / bl.z) * np.exp(exponent)

	error_count = (y_true == y_hypoth).astype(int)
	print(np.unique(error_count))
	assert error_count.min() == 0
	assert error_count.max() == 1 

	return new_weights, error_count.mean()



def do_boosting(data, cur_weights, cur_hypoth=None, T=0):
	'''
	Given a dataset and some weights, constructs an ensemble hypothesis
	based on a linear comb of weak-classifiers

	Inputs:
		data: (np array) of images
		cur_weights: (np array) weights of observations
		cur_hypoth: (list of WeakClassifiers)
		T: (int) count of boosting rounds
	'''

	if cur_hypoth is None:
		cur_hypoth = []

	weak_learner = best_learner(cur_weights)
	error = weak_learner.error 

	alpha = (1/2) * np.log( (1-error) / error )
	weak_learner.alpha = alpha

	cur_hypoth.append(weak_learner)

	z = 2 * np.sqrt( (error * (1-error) ) )
	weak_learner.z = z

	new_weights, error_rate = update_weights(cur_weights, weak_learner)
	print(weak_learner)
	print("QC'ed error rate=", error_rate)

	# recursive call
	if T < 5:
		return do_boosting("stand_in", new_weights, cur_hypoth, T+1)

	else:
		return cur_hypoth

# Define a filter's starting position and determine the reuslting coordinates
filter1 = [0,0,2,2,2,0,4,2]  # use stride 2 or 1
filter2 = [0,0,1,1,1,0,2,1]  # use stride 1
filter3 = [0,0,4,4,4,0,8,4]  # use stride 1
# filter1_coords = new_features(D, stride=2, primitive=filter1)

# # define the global II table for lookups
II_TABLES = load_training(TRAINING_SIZE)

# # define the global for all coordinates for our features
FEATURE_COORDS = new_features(D, stride=1, primitive=filter3)

IS_IMAGE = np.array( [1]*TRAINING_SIZE + [0]*TRAINING_SIZE)
IS_BACKGROUND = np.array( [0]*TRAINING_SIZE + [1]*TRAINING_SIZE)
init_weights = np.full( (2*TRAINING_SIZE), 1/(2*TRAINING_SIZE) )

hypothesis = do_boosting("stand_in", init_weights)



### BELOW THIS IS TESTING CODE

'''
test_image = np.array( [ [1]*4, [2]*4, [3]*4, [4]*4 ] )
test_II = test_image.cumsum(axis=0).cumsum(axis=1)

II_TABLES = np.array([test_II])
FEATURE_COORDS = new_features(4, stride=1, primitive=filter1)
ft_count = len(FEATURE_COORDS)

for ft_num in range(ft_count):
	f = FEATURE_COORDS[ft_num]
	print("Coords: ", f[0:4], "to", f[4:8])
	val = compute_feature(0, ft_num)
	print("\t val = ", val)
	print()


#II_TABLES = load_training(TRAINING_SIZE)
#FEATURE_COORDS = define_features(D, STRIDE, SCALE_FILTERS)
WEIGHTS = np.full( (2*TRAINING_SIZE), 1/(2*TRAINING_SIZE) )
IS_IMAGE = np.array( [1]*TRAINING_SIZE + [0]*TRAINING_SIZE)
IS_BACKGROUND = np.array( [0]*TRAINING_SIZE + [1]*TRAINING_SIZE)
'''
