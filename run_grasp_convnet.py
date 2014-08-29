import cPickle
import numpy as np
import theano
import theano.tensor as T
import os
import h5py
import collections
import pylab


import pylearn2
import matplotlib.pyplot as plt

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

out = feature_extractor(img)
out_rolled = np.rollaxis(out,1,4)
out_window = out_rolled[0,:,:,:]

f = open(CONV_MODEL_FILENAME)
cnn_model = cPickle.load(f)

cnn_model = cnn_model.layers[-1]

W = cnn_model.get_weights_topo()
W = W[0,0,:,:]

output = np.dot(out_window,W) + cnn_model.b.get_value()
import IPython
IPython.embed()

#f(np.zeros((4, 240, 240, 1), dtype=np.float32))
