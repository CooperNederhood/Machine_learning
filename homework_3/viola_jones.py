import numpy as np 
import numpy.linalg as linalg
import skimage.color as skimage
import PIL 

D = 64
TRAINING_SIZE = 2000
STRIDE = 1
SCALE_FILTERS = 1

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

#II_TABLES = load_training(TRAINING_SIZE)

def define_features(dimension, stride, scale):
	'''
	Creates numpy array of dimension
	feature_count x 2 x 2. Assumes 2 rectangle
	features

	NOTE: filter_coords has black (+) coords first,
			then lwhite (-) coords next
	'''

	# 2 x 1 filter

	# define rect boundaries for 2x1 
	p0 = [0,0]
	p1 = [1*scale, 1*scale]
	q0 = [1*scale, 0]
	q1 = [2*scale, 1*scale]
	points = [p0, p1, q0, q1]

	filter_coords = []
	filter_coords.append( p0+p1+q0+q1 )

	print("p0 initialized: {}".format(p0))
	print("p1 initialized: {}".format(p1))
	print("q0 initialized: {}".format(q0))
	print("q1 initialized: {}".format(q1))
	print("")

	while q1 != [dimension,dimension]:

		# try to go right
		if p1[1] != dimension:
			for pt in points:
				pt[1] += stride

		# if not, then we need to reset and move down
		else:
			p0[1] = 0
			p1[1] = 1*scale
			q0[1] = 0
			q1[1] = 1*scale

			for pt in points:
				pt[0] += stride
		print("p0 is: {}".format(p0))
		print("p1 is: {}".format(p1))
		print("q0 is: {}".format(q0))
		print("q1 is: {}".format(q1))
		print("")

		# Add points to the filter_coords list
		filter_coords.append( p0+p1+q0+q1 )

	return filter_coords

test_coords = define_features(4, 2, 2)