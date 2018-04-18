import numpy as np
import csv
import matplotlib.pyplot as plt


TRAIN_PATH = './train.csv'
TEST_PATH = './test.csv'
OUTPUT_PATH = './predictions.csv'
HEIGHT = 28
WIDTH = 28
CLASS_NUM = 10



def read_training_set():
	"""
	returns: X - training set (None, 28*28) FOR KERAS
			 Y - one hot labels of X (None, 10)
	"""
	f = open(TRAIN_PATH)
	train = csv.reader(f)

	X = []
	Y = []

	for t in train:
		break  # drop the 1st line

	for t in train:
		Y.append(int(t[0]))
		X.append(np.array(t[1:], dtype=int))
		
	# X = np.transpose(X)
	Y = num_to_one_hot(Y, CLASS_NUM)
	X = np.array(X)
	# Y = np.array(Y)

	f.close()
	return X, Y


def read_test_set():
	"""
	returns: X - test set (None, 28*28) FOR KERAS
	"""
	f = open(TEST_PATH)
	test = csv.reader(f)

	X = []

	for t in test:
		break  # drop the 1st line

	for t in test:
		X.append(np.array(t, dtype=int))
		
	# X = np.transpose(X)
	X = np.array(X)

	f.close()
	return X


def num_to_one_hot(Y, num_classes):
	"""
	args: Y - numerical labels
		  num_classes - #classes

	returns: output - one hot labels of X (None, 10)
	"""
	width = num_classes
	height = len(Y)
	
	output = np.zeros((height, width))
	
	for i in range(height):
		output[i, Y[i]] = 1
		
	return output


def plot_image(array):
	"""
	plot an image
	"""
	img = np.reshape(array, (HEIGHT, WIDTH))
	plt.imshow(img)


def write_predictions(pred):
	"""
	output the predictions as a .csv file
	"""
	with open(OUTPUT_PATH, "w", newline='') as csvfile: 
	    output = csv.writer(csvfile)
	    output.writerow(['ImageId', 'Label'])
	    for i in range(len(pred)):
	    	output.writerow([str(i+1), str(pred[i])])
	return output