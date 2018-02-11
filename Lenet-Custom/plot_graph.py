# file kfkd.py
try:
    import cPickle as pickle
except ImportError:  # Python 3
    import pickle
from datetime import datetime
import os
import sys

from matplotlib import pyplot
import numpy as np
from lasagne import layers
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet
from pandas import DataFrame
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import theano

import cv2

try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
except ImportError:
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer


FTRAIN = '/home/sprva/AAkash/TUM/Deeplearning-Project/kaggle_facial_dataset/training.csv'
FTEST = '/home/sprva/AAkash/TUM/Deeplearning-Project/kaggle_facial_dataset/test.csv'
FLOOKUP = '/home/sprva/AAkash/TUM/Deeplearning-Project/kaggle_facial_dataset/IdLookupTable.csv'


class FlipBatchIterator(BatchIterator):
    flip_indices = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),
        ]

    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        if yb is not None:
            # Horizontal flip of all x coordinates:
            yb[indices, ::2] = yb[indices, ::2] * -1

            # Swap places, e.g. left_eye_center_x -> right_eye_center_x
            for a, b in self.flip_indices:
                yb[indices, a], yb[indices, b] = (
                    yb[indices, b], yb[indices, a])

        return Xb, yb


class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.cast['float32'](self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()


def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y


def plot_sample(x, y, axis, original_img):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', c='r', s=100)
    print(y.size)

    #f = open('datapoints.txt', "w")	
   
    counter = 0
    output_file = ""
    font = cv2.FONT_HERSHEY_SIMPLEX

    original_img = cv2.resize(original_img, (200, 200)) 
    scale = 200/96

    while counter < y.size:
		cv2.circle(original_img, (int((y[counter]* 48 + 48)*scale), int((y[counter+1]* 48 + 48)*scale)), 1, (255,0,0), 2)
		cv2.putText(original_img, str((counter/2)+1), (int((y[counter]* 48 + 48)*scale - 3), int((y[counter+1]* 48 + 48)*scale - 5)), font, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
		output_file = output_file + str(int(y[counter]* 48 + 48)) + " " + str(int(y[counter+1]* 48 + 48)) + " " 
		counter = counter + 2

    #f.write(output_file)
    cv2.imshow('Draw01',original_img)
    cv2.imwrite("side_output.jpg",original_img)

    cv2.waitKey(10)	   
	    
    #f.close

'''
img = cv2.imread('side_face.jpeg')
original_img = img
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (96, 96)) 

data = np.array(gray)
flattened = data.flatten()

print data.shape
print flattened.shape

X1 = np.vstack(flattened) / 255.  # scale pixel values to [0, 1]
X1 = X1.astype(np.float32)
X1 = X1.reshape(1, 1, 96, 96)
print(X1.shape)


#X, _ = load(test=True)
#print(X.shape)
#X = X.reshape(-1, 1, 96, 96)
#print(X.shape)

with open('net.pickle', 'rb') as f:
    net = pickle.load(f)
y_pred = net.predict(X1)

'''

df = read_csv("NMSE_custom.csv")  # load pandas dataframe
df1 = read_csv("NMSE_hyperface.csv")  # load pandas dataframe


# The Image column has pixel values separated by space; convert
# the values to numpy arrays:
#df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

#if cols:  # get a subset of columns
#    df = df[list(cols) + ['Image']]

#print(df.count())  # prints the number of values for each column
#df = df.dropna()  # drop all rows that have missing values in them

X = np.vstack(df['NMSE_custom'].values)  # scale pixel values to [0, 1]
Y = np.vstack(df1['NMSE_hyperface'].values)  # scale pixel values to [0, 1]

#X = X.astype(np.float32)

'''
if not test:  # only FTRAIN has any target columns
    y = df[df.columns[:-1]].values
    y = (y - 48) / 48  # scale target coordinates to [-1, 1]
    X, y = shuffle(X, y, random_state=42)  # shuffle train data
    y = y.astype(np.float32)
else:
    y = None

return X, y
'''
#train_loss = np.array([i["train_loss"] for i in net.train_history_])
#valid_loss = np.array([i["valid_loss"] for i in net.train_history_])
pyplot.plot(X, linewidth=3, label="NMSE_custom")
pyplot.plot(Y, linewidth=3, label="NMSE_hyperface")
pyplot.grid()
pyplot.legend()
pyplot.xlabel("test_images")
pyplot.ylabel("NMSE")
#pyplot.ylim(1e-3, 1e-2)
pyplot.ylim(0, 1)
#pyplot.yscale("log")
pyplot.show()
'''
print(y_pred.shape)

fig = pyplot.figure(figsize=(96, 96))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(1):
    ax = fig.add_subplot(1, 1, i + 1, xticks=[], yticks=[])
    plot_sample(X1[i], y_pred[i], ax, original_img)

pyplot.show()
'''