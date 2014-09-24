import cPickle
import numpy as np
import theano
import theano.tensor as T
import os
import h5py
import collections
import pylab
from scipy.signal import argrelextrema
import scipy.misc
import time

import pylearn2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


PYLEARN_DATA_PATH = os.environ["PYLEARN2_DATA_PATH"]


CONV_MODEL_FILENAME = 'models/grasp_72x72_model/cnn_model.pkl'


def run_72X72():
    f = open(CONV_MODEL_FILENAME)
    cnn_model = cPickle.load(f)
    X = cnn_model.get_input_space().make_theano_batch()

    Y = cnn_model.fprop(X)

    f = theano.function([X], Y)

    f(np.zeros((4, 72, 72, 1), dtype=np.float32))


def get_feature_extractor():
    f = open(CONV_MODEL_FILENAME)
    cnn_model = cPickle.load(f)
    new_space = pylearn2.space.Conv2DSpace((240, 320), num_channels=4, axes=('c', 0, 1, 'b'), dtype='float32')

    cnn_model.layers = cnn_model.layers[0:-1]

    #we want to padd zeros around the edges rather than ignoring edge pixels
    for i in range(len(cnn_model.layers)):
        cnn_model.layers[i].border_mode = "full"

    cnn_model.set_input_space(new_space)

    X = cnn_model.get_input_space().make_theano_batch()

    Y = cnn_model.fprop(X)

    feature_extractor = theano.function([X], Y)
    return feature_extractor

def get_classifier():
    f = open(CONV_MODEL_FILENAME)
    cnn_model = cPickle.load(f)

    cnn_model = cnn_model.layers[-1]

    X = cnn_model.get_input_space().make_theano_batch()

    Y = cnn_model.fprop(X)

    classifier = theano.function([X], Y)
    return classifier



def get_local_extrema(output):
    output2 = np.copy(output)*-1
    e = np.zeros(output2.shape)
    extrema = argrelextrema(output2, np.greater)
    for i in range(len(extrema[0])):
        e[extrema[0][i], extrema[1][i], extrema[2][i]] = output[extrema[0][i], extrema[1][i], extrema[2][i]]

    return e


def get_scaled_image(img, scale=(34,44)):
    return scipy.misc.imresize(img, scale)



# f = open(CONV_MODEL_FILENAME)
# cnn_model = cPickle.load(f)
# new_space = pylearn2.space.Conv2DSpace((240, 240), num_channels=4, axes=('c', 0, 1, 'b'), dtype='float32')
# cnn_model.set_input_space(new_space)
#
# X = cnn_model.get_input_space().make_theano_batch()
#
# Y = cnn_model.fprop(X)
#
# f = theano.function([X], Y)
feature_extractor = get_feature_extractor()
img = np.zeros((4,240,320,1),dtype=np.float32)
#dataset = h5py.File('/home/jvarley/grasp_deep_learning/data/rgbd_images/coke_can/rgbd_and_labels.h5')
#img_in = dataset['rgbd'][0,240-120:240+120,320-160:320+160,:]


dataset = h5py.File('/home/jvarley/grasp_deep_learning/data/rgbd_images/saxena_partial_rgbd_and_labels.h5')


plt.imshow(dataset['rgbd_data'][0, :,:,0:3])
plt.show()

img_in = dataset['rgbd_data'][0,320-120:320+120,320-160:320+160,:]

f = open(CONV_MODEL_FILENAME)
cnn_model = cPickle.load(f)

cnn_model = cnn_model.layers[-1]

W = cnn_model.get_weights_topo()
W = W[0, 0, :, :]

start = time.time()
img[:, :, :, 0] = np.rollaxis(img_in, 2, 0)
print time.time()-start

start = time.time()
out = feature_extractor(img)
print time.time()-start

start = time.time()
out_rolled = np.rollaxis(out,1,4)
out_window = out_rolled[0,:,:,:]
output = np.dot(out_window,W) + cnn_model.b.get_value()
print time.time()-start

#print time.time()-start

#currently output.min() is < 0 and output.max() > 0
#normalized between 0 and 255
output = 255 - (output-output.min())/(output.max()-output.min())*255.0

#crop output since padding messes up the borders
border_dim = 5
output = output[border_dim:-border_dim, border_dim:-border_dim]


#print time.time()-start

outdir = "/home/jvarley/grasp_deep_learning/data/final_output/"


print "plot output"
fig = plt.figure()
ax = plt.axes(projection='3d')

x = np.zeros(output.shape[0]*output.shape[1])
y = np.zeros(output.shape[0]*output.shape[1])
z = np.zeros(output.shape[0]*output.shape[1])
count = 0
for i in range(output.shape[0]):
    for j in range(output.shape[1]):
        x[count] = i
        y[count] = j
        z[count] = output[i, j, 0]
        count += 1

ax.scatter(x, y, z)
#plt.show()
plt.savefig(outdir + "abc1.png")

plt.close()

print "input: "
plt.imshow(img_in)
plt.show()
scipy.misc.imsave(outdir + "zyx.png", img_in[:,:,0:3])


print "output.shape: " + str(output.shape)
plt.imshow(output[:, :, 0])
plt.show()

extremas = get_local_extrema(output)
print "extremas:"
plt.imshow(extremas[:, :, 0])
plt.show()

print "extrema imposed on output: "
plt.imshow(output[:, :, 0] + (extremas[:, :, 0]))
plt.show()

print "extrema imposed on input: "
plt.imshow((get_scaled_image(img_in) + extremas)/(get_scaled_image(img_in) + extremas).max())
plt.show()

