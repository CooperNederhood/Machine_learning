import numpy as np 
import numpy.linalg as linalg
import skimage.color as skimage
import PIL
from PIL import Image, ImageDraw 

D = 64
TRAINING_SIZE = 100

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
		str_3 = "\t polarity={}\n".format(self.polarity)
		str_4 = "\t error={}\n".format(self.error)

		return str_1+str_2+str_3+str_4

	def calc_hypoth(self, data):
		'''
		Returns an array of the predicted y-values
		for the given data
		'''

		total_pic_amount = data.shape[0]

		ft_vals = np.empty( total_pic_amount )

		for pic_num in range(total_pic_amount):
			ft_vals[pic_num] = compute_feature(data, pic_num, self.i)

		### Add a negative before polarity bc I think mine is backwards
			# may want to change this at some point upstream of calcuations
		y_hypoth = np.sign(-self.polarity * (ft_vals - self.theta))
		y_hypoth[ y_hypoth==0 ] = 1

		return y_hypoth

	def calc_f_vals(self, data):
		'''
		Returns an array of the function-values
		for the given data
		'''

		total_pic_amount = data.shape[0]

		ft_vals = np.empty( total_pic_amount )

		for pic_num in range(total_pic_amount):
			ft_vals[pic_num] = compute_feature(data, pic_num, self.i)

		f_vals = -self.polarity * (ft_vals - self.theta)

		return f_vals


class BoostedLearner:
	""" Represents a single boosted hypothesis, stored as a list of WeakLearners.
	Essentially a storage class with methods to determine the threshold big_theta
	and another to determine error rate
	"""

	def __init__(self):
		'''
		wk_list is the list of weak learners making up the boosted learner
		big_theta is the threshold to force zero false negatives
		'''

		self.wk_list = []
		self.big_theta = 0
		self.predictions = None
		self.false_pos_rate = None
		self.fase_neg_rate = None

		self.pred_face_avg = None
		self.pred_back_avg = None

	def add_weak_classifier(self, wk):
		'''
		Adds weak classifer to boosted learner
		'''
		self.wk_list.append(wk)

	def calc_f_vals(self, data):
		'''
		Helper method to calculate the f(x) values for 
		the specified data
		
		Inputs:
			- data: (np array) of II for current set of images

		Returns:
			- f_data: (np array) of f(x) values for all images in data
		'''

		image_count = data.shape[0]
		f_data = np.zeros(image_count)

		for weak_c in self.wk_list:
			#weighted_pred_y = (weak_c.alpha) * weak_c.calc_hypoth(data)
			weighted_pred_y = (weak_c.alpha) * weak_c.calc_f_vals(data)
			f_data += weighted_pred_y

		return f_data 


	def reset_big_theta(self, data, image_type_flags):
		'''
		Resets and returns the big_theta param which forces zero
		false negatives

		Inputs:
			- data: (np array) of II for current set of images
			- image_type_flags: (2 lists) indicating the image/background for
									the current images still in data
		'''
		#unpack our image/background indicators
		is_image = image_type_flags[0]
		is_background = image_type_flags[1]

		boosting_depth = len(self.wk_list)

		f_data = self.calc_f_vals(data)

		# we have the f(x_i) values for all images i
		# zero out if it's a background image
		f_data_faces = f_data * is_image

		min_f = f_data_faces.min()

		if min_f < 0:
			self.big_theta = np.abs(min_f)
		else:
			self.big_theta = 0

		return self.big_theta

	def make_prediction(self, data):
		'''
		Makes y_hat prediction based on weak learners and the 
		current big_theta cutoff

		Returns:
			- y_pred: predicting cat (+1 is image; -1 is background)
		'''
		f_data = self.calc_f_vals(data)
		#f_plus_theta = f_data
		f_plus_theta = f_data + self.big_theta

		y_pred = np.sign(f_plus_theta)
		y_pred[ y_pred==0 ] = 1

		self.predictions = y_pred
		self.pred_face_count = y_pred[ y_pred == 1].size
		self.pred_back_count = y_pred[ y_pred == -1].size

		self.pred_face_avg = f_data[ y_pred == 1].mean()
		self.pred_back_avg = f_data[ y_pred == -1].mean()

		return y_pred

	def set_error_rates(self, data, image_type_flags):
		'''
		Calculates the boosted learner false positive rate

		Inputs:
			- data: (np array) of II for current set of images
			- image_type_flags: (2 lists) indicating the image/background for
									the current images still in data
		'''
		#unpack our image/background indicators
		is_image = image_type_flags[0]
		is_background = image_type_flags[1]

		y_true = is_image - is_background

		y_pred = self.make_prediction(data)
		errors = (y_true != y_pred).astype(int)

		bool_is_image = is_image == 1
		bool_is_background = is_background == 1

		false_neg = errors[y_pred == -1]
		self.false_neg_rate = false_neg.mean()

		false_pos = errors[y_pred == 1]
		self.false_pos_rate = false_pos.mean()


	def pretty_print(self, details=False):

		print("Boosted Learner has:\n")
		print("\t big_theta={}\n".format(self.big_theta))
		print("\t predicted faces={}\n".format(self.pred_face_count))
		print("\t predicted avg fn value={}\n".format(self.pred_face_avg))

		print("\t predicted backgrounds={}\n".format(self.pred_back_count))
		print("\t predicted avg fn value={}\n".format(self.pred_back_avg))

		print("\t false pos rate={}\n".format(self.false_pos_rate))
		print("\t false neg rate={}\n".format(self.false_neg_rate))

		if details:
			for weak_l in self.wk_list:
				print(weak_l)


class CascadeClassifier():
	"""
	Reprsents a Viola-Jones style cascade classifer, 
	composed of a sequence of BoostedLearners
	"""

	def __init__(self, orig_data, orig_flags, structure):
		'''
		Attributes:
			- orig_data: (np array) of II tables to train on
			- orig_flags: (np array) of face/no face flags
			- structure: (list) of the max boosting rounds for each cascade 
		'''
		
		self.orig_data = orig_data 
		self.orig_flags = orig_flags
		#self.max_boosting_depth = max_boosting_depth
		self.structure = structure
		
		self.cascade_depth = len(structure)
		self.booster_list = []
		self.current_data = orig_data
		self.current_flags = orig_flags 

	def add_booster(self, data, image_type_flags, max_depth, false_pos):
		'''
		Adds a booster classifer to the booster_list and 
		returns the array of photo observations from data which
		should be removed because they are predicted to be non-faces

		Inputs:
			data: (np array) of II for the current training set
			image_type_flags: (2 lists) indicating the image/background for
									the current images still in data
			max_depth: (int) of max amount of weak classifiers to include
			false_pos: (float) max false positive rate

		Returns:
			restricted_data, restricted_image_flags - removes non-face photos
		'''

		picture_count = data.shape[0]

		init_weights = np.full( (picture_count), 1/(picture_count) )
		booster = do_boosting(data, init_weights, image_type_flags, max_depth, fp_rate=false_pos)[1]

		self.booster_list.append(booster)

		predictions = booster.make_prediction(data)
		pred_photo = predictions == 1

		restricted_data = data[ pred_photo ]

		# unpack flags to then restrict and repack
		init_is_image, init_is_background = image_type_flags

		init_flags = (init_is_image, init_is_background)
		restricted_is_image = init_is_image[ pred_photo ]
		restricted_is_background = init_is_background[ pred_photo ]
		restricted_image_flags = [restricted_is_image, restricted_is_background]

		cur_count = data.shape[0]
		restricted_count = restricted_data.shape[0]
		print("Had {} images now remove {} image.....".format(cur_count, cur_count-restricted_count))
		print()

		return restricted_data, restricted_image_flags

	def build_classifier(self):
		'''
		Builds a cascading classifier
		'''

		# Add layers
		for cur_cascade, s in enumerate(self.structure):
			max_depth = s[0]
			false_pos = s[1]
			print("#####################################")
			print("BEGIN CASCADE (round = {})".format( cur_cascade+1))
			print("PICTURES REMAINING = ", self.current_data.shape[0])
			new_data, new_flags = self.add_booster(self.current_data, self.current_flags, max_depth, false_pos)
			self.current_data = new_data
			self.current_flags = new_flags 



	def pretty_print(self):

		for booster in self.booster_list:
			booster.pretty_print()


	def make_prediction(self, test_data):
		'''
		Given a table of II, returns list of which
		picture numbers in the test data contain faces
		'''

		num_test_images = test_data.shape[0]
		picture_numbers = np.array(range(num_test_images))

		for boosted_learner in self.booster_list:

			preds = boosted_learner.make_prediction(test_data)
			is_face = preds == 1

			picture_numbers = picture_numbers[is_face]
			test_data = test_data[is_face]

			if len(picture_numbers) == 0:
				break 


		return picture_numbers




def load_data(size, start_at=0):
	'''
	Given an integer of the size of the data in 
	each category, loads data, and returns 
	II-lookup table

	Converts images to grayscale

	Inputs: 
		- size (int): of how many background and faces to return
		- start_at (int): start at photo 0, or further into data

	Returns: ii_table (np array dim: size*2 x 64 x 64)
	'''

	raw_data = np.empty( (size*2, D, D) )
	ii_table = np.empty( (size*2, D, D) )

	# background images - coded as negatives
	for i in range(size):
		im = PIL.Image.open("background/{}.jpg".format(i))
		grey_im = skimage.rgb2grey( np.array(im) )
		assert grey_im.shape == (D, D)

		cum_sum = grey_im.cumsum(axis=0).cumsum(axis=1)

		ii_table[i] = cum_sum
		raw_data[i] = np.array(grey_im)

	# face images - coded as positives
	for i in range(size):
		im = PIL.Image.open("faces/face{}.jpg".format(i))
		grey_im = skimage.rgb2grey( np.array(im) )
		assert grey_im.shape == (D, D)

		cum_sum = grey_im.cumsum(axis=0).cumsum(axis=1)

		ii_table[i+size] = cum_sum
		raw_data[i+size] = np.array(grey_im)

	# is_image = np.array( [1]*size + [0]*size)
	# is_background = np.array( [0]*size + [1]*size)
	is_background = np.array( [1]*size + [0]*size)
	is_image = np.array( [0]*size + [1]*size)

	#return ii_table, raw_data
	return ii_table, is_image, is_background, raw_data


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

		# Add points to the filter_coords list
		filter_coords.append( p0+p1_ex+q0+q1_ex )

	return np.array(filter_coords)



def do_test(file, trained_classifer):

	im = Image.open(file)
	draw = ImageDraw.Draw(im)
	size = 64
	count = 0

	faces = []

	for x in range(64, im.size[0]+1):
		for y in range(64, im.size[1]+1):

			sub = im.crop((x-64, y-64, x, y))
			count +=1 
			photo_array = np.array(sub)
			cum_sum = photo_array.cumsum(axis=0).cumsum(axis=1)
			ii_table = np.array( [cum_sum] )
			assert ii_table.shape == (1, 64, 64)

			# predict
			pred = trained_classifer.make_prediction(ii_table)
			if len(pred) == 1:
				faces.append(sub)
				draw.rectangle((x-64, y-64, x, y), outline=400)

			#print("Subphoto from {}, {}, {}, {} has size = {}".format(x-64,y-64,x,y, sub.size))
			#print("\tPrediction is:", pred)

		if np.mod(x,10) == 0:
			print("Testing subphoto at {}, {}, {}, {}".format(x-64,y-64,x,y))

			
	return count, faces, im




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

def compute_feature(data, pic_number, ft_number):
	'''
	Computes the value of feature # ft_number on
	image # pic_number

	Inputs:
		- data: (np array) of II for current set of images
	'''

	# Get the respective ii_table and feature coords from our globals
	ii_table = data[pic_number]
	ft_coords = FEATURE_COORDS[ft_number]

	pos_weight = return_II(ii_table, ft_coords[0:2], ft_coords[2:4])
	neg_weight = return_II(ii_table, ft_coords[4:6], ft_coords[6:8])

	return pos_weight - neg_weight

def find_p_theta(data, image_type_flags, ft_number, cur_weights, verbose=False):
	'''
	Before we can find the best weak learner, for each feature we must find
	the optimal cutoff theta and the polarity, p, which defines the best 
	weak learner for this featuer

	has_image denotes positive polarity

	Inputs:
		- data: (np array) of II for current set of images
		- image_type_flags: (2 lists) indicating the image/background for
								the current images still in data
		- ft_number: (int) indicating the feature number to evaluate
		- cur_weights: (np array) weight assigned to each 

	Returns:
		- weak_learner: (WeakLearner type) containing ft_number, theta, pol, error
	'''

	pic_count = data.shape[0]

	ft_vals = np.empty( pic_count )

	for pic_num in range(pic_count):
		ft_vals[pic_num] = compute_feature(data, pic_num, ft_number)

	ranking = np.argsort(ft_vals)

	#unpack our image/background indicators
	is_image = image_type_flags[0]
	is_background = image_type_flags[1]

	sorted_weights = cur_weights[ranking]
	sorted_is_image = is_image[ranking]
	sorted_is_background = is_background[ranking]
	sorted_ft_vals = ft_vals[ranking]

	cum_image = np.cumsum(sorted_weights*sorted_is_image)
	cum_background = np.cumsum(sorted_weights*sorted_is_background)

	T_img = cum_image[-1]
	T_back = cum_background[-1]

	e = np.fmin(cum_image + (T_back - cum_background), cum_background + (T_img - cum_image) )

	min_e = np.argmin(e)
	minimum_error = e[min_e]

	#assert minimum_error > 0 and minimum_error <= .50
	assert minimum_error >= 0 and minimum_error <= .50

	if minimum_error > .5:
		print("Feature #{} has min error of {}".format(ft_number, minimum_error))

	if min_e == pic_count -1:
		theta = sorted_ft_vals[min_e]
	else:
		theta = (sorted_ft_vals[min_e] + sorted_ft_vals[min_e+1] ) / 2

	polarity = 1 if cum_image[min_e] + (T_back - cum_background[min_e]) > cum_background[min_e] + (T_img - cum_image[min_e]) else -1

	weak_learner = WeakLearner(ft_number, theta, polarity)
	weak_learner.error = minimum_error

	if verbose:
		return weak_learner, ft_vals, ranking, sorted_weights

	else:
		return weak_learner


def best_learner(data, cur_weights, image_type_flags):
	'''
	Given the current weights, returns the i-feature #, the polarity, and theta-cutoff
	which minimizes the error. i.e. this is our best weak classifier

	Inputs:
		data: (np array) of II for the current training set
		cur_weights: (np array) weights of observations
		image_type_flags: (2 lists) indicating the image/background for
								the current images still in data

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

		cur_learner = find_p_theta(data, image_type_flags, ft_num, cur_weights)

		#print(cur_learner)
		#print("")

		if cur_learner.error < min_error:
			opt_learner = cur_learner
			min_error = cur_learner.error

			#print("Feature #{} is NEW BEST CLASSIFIER".format(cur_learner.i))
			
	return opt_learner


def update_weights(cur_weights, bl, data, image_type_flags):
	'''
	Given the current set of weights and the new weak
	learner to be added to the boosting hypothesis,
	returns the re-calculated weights 

	Inputs:
		- cur_weights: current weights of the training pop
		- bl: (WeakLearner) best weak learner in this boosting round
		- data: (np array) of II for the current training set
		- image_type_flags: (2 lists) indicating the image/background for
								the current images still in data

	Returns:
		- new_weights (np array), unweighted error of weak-learner
	'''
	#unpack our image/background indicators
	is_image = image_type_flags[0]
	is_background = image_type_flags[1]

	y_true = is_image - is_background
	y_hypoth = bl.calc_hypoth(data)

	exponent =  -bl.alpha * y_true * y_hypoth

	new_weights = (cur_weights ) * np.exp(exponent)
	new_weights = new_weights / new_weights.sum()
	if np.abs(new_weights.sum() - 1) > 0.001:
		print("WEIGHTS != 1")
		return new_weights, cur_weights

	error_count = (y_true != y_hypoth).astype(int)

	weighted_error_count = error_count*cur_weights

	print(np.unique(error_count))
	assert error_count.min() == 0
	assert error_count.max() == 1 

	return new_weights, weighted_error_count.sum()



def do_boosting(data, cur_weights, image_type_flags, T, fp_rate, cur_hypoth=None, cur_T=1):
	'''
	Given a dataset and some weights, constructs an ensemble hypothesis
	based on a linear comb of weak-classifiers

	Inputs:
		data: (np array) of II for the current training set
		cur_weights: (np array) weights of observations
		image_type_flags: (2 lists) indicating the image/background for
								the current images still in data
		T: (int) rounds of boosting
		fp_rate: (float) false positive cutoff
		cur_hypoth: (list of WeakClassifiers)
		cur_T: (int) cuurent count of boosting rounds
	'''

	print("Boosting round {} of {}".format(cur_T, T))
	if cur_hypoth is None:
		cur_hypoth = BoostedLearner()

	weak_learner = best_learner(data, cur_weights, image_type_flags)
	print(weak_learner)
	error = weak_learner.error 

	alpha = (1/2) * np.log( (1-error) / error )
	weak_learner.alpha = alpha

	cur_hypoth.add_weak_classifier(weak_learner)

	z = 2 * np.sqrt( (error * (1-error) ) )
	weak_learner.z = z

	new_weights, error_rate = update_weights(cur_weights, weak_learner, data, image_type_flags)

	# reset big_theta of our cur_hypothesis
	big_theta = cur_hypoth.reset_big_theta(data, image_type_flags)
	#cur_hypoth.big_theta = 0

	# check the error of our cur_hypothesis
	cur_hypoth.set_error_rates(data, image_type_flags)


	cur_hypoth.pretty_print()


	if cur_T < T and cur_hypoth.false_pos_rate > fp_rate:
		return do_boosting(data, new_weights, image_type_flags, T, fp_rate, cur_hypoth, cur_T+1)
	else:
		return new_weights, cur_hypoth



# Define a filter's starting position and determine the reuslting coordinates
filter1 = [0,0,8,8,8,0,16,8]  # use stride 2 or 1
features1 = new_features(D, stride=2, primitive=filter1)

# filter2 = [0,0,1,1,1,0,2,1]  # use stride 1

# filter3 = [0,0,4,4,4,0,8,4]  # use stride 1
# features3 = new_features(D, stride=1, primitive=filter3)

# filter4 = [0,0,2,8,2,0,4,8]
# features4 = new_features(D, stride=2, primitive=filter4)

filter2 = [0,0,8,8,0,8,8,16]  # use stride 2 or 1
features2 = new_features(D, stride=2, primitive=filter2)

# Define the initial data before cascading and removing
init_ii_tables, init_is_image, init_is_background, raw_data = load_data(TRAINING_SIZE, 0)
init_weights = np.full( (2*TRAINING_SIZE), 1/(2*TRAINING_SIZE) )
init_flags = (init_is_image, init_is_background)

# # define the global for all coordinates for our features
#FEATURE_COORDS = np.concatenate( (features1, features3, features4), axis=0)
FEATURE_COORDS = np.concatenate( (features1, features2), axis=0)

#vg0_structure = [1, 2, 5]
vg0_structure = [(1, .3) , (5, .2), (5, .1), (10, 0.075)]
vg0 = CascadeClassifier(orig_data = init_ii_tables, orig_flags = init_flags, structure=vg0_structure)
vg0.build_classifier()

'''
iters, results, im = do_test("small_test.jpg", vg0)
im.show()
'''
