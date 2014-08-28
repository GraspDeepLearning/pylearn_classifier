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
import slic_image

PYLEARN_DATA_PATH = os.environ["PYLEARN2_DATA_PATH"]

NYU_DEPTH_V2_PATH = PYLEARN_DATA_PATH + "/nyu_depth_labeled/nyu_depth_v2_labeled.mat"
NYU_DEPTH_V2_DATASET = h5py.File(NYU_DEPTH_V2_PATH)

CONV_MODEL_FILENAME = 'models/nyu_72x72_model/convolutional_network_best2.pkl'



class DenseRGBDClassifier():

    def __init__(self, conv_model_filepath):

        self.conv_model_filepath = conv_model_filepath

        self.classifier = self.get_classifier()
        self.feature_extractor = self.get_feature_extractor()

    #this is the top half of the trained convnet
    #we need to remove the softmax functionality since
    # 1) theano does not support softmax on a 3d tensor
    # 2) we only need it for training, we can run argmax on the
    #    raw outputs of the classifier without running softmax first
    def get_classifier(self):
        f = open(self.conv_model_filepath)
        cnn_model = cPickle.load(f)
        cnn_model = cnn_model.layers[-1]

        #This is the output of the feature extractor
        #shape should be:
        #(~num_rows, ~num_cols, num_features)
        #it is ~num_rows and ~num_cols because depending on
        #pool_stride and kernel_size from the feature extraction, there is some 0-padding of the input
        # and the max pooling with scale the output down
        X = cnn_model.get_input_space().make_theano_batch()

        #Z is now a 3d tensor of shape
        # (~num_rows,~num_cols, num_labels)
        Y = T.dot(X, cnn_model.W) + cnn_model.b

        #we can take the arg max to get the per pixel label
        #So Y is of shape:
        # (1, ~num_rows, ~num_cols)
        Y_out = T.argmax(Y, axis=3)

        return theano.function([X], Y_out)

    #this is the bottom half of the trained conv net
    #we just change the shape of the input and remove the
    # fully connected top layer
    def get_feature_extractor(self):
        f = open(self.conv_model_filepath)
        cnn_model = cPickle.load(f)

        #we want the input space to be the entire image
        new_space = pylearn2.space.Conv2DSpace((320, 240), num_channels=4, axes=('b', 0, 1, 'c'), dtype='float32')

        #we need to get rid of the softmax layer so that
        #we return a result for each pixel rather than
        #the best category for the entire image
        cnn_model.layers = cnn_model.layers[0:-1]

        #we want to padd zeros around the edges rather than ignoring edge pixels
        for i in range(len(cnn_model.layers)):
            cnn_model.layers[i].border_mode = "full"

        cnn_model.set_input_space(new_space)
        X = cnn_model.get_input_space().make_theano_batch()
        Y = cnn_model.fprop(X)

        f = theano.function([X], Y)
        return f

    def run(self, image):

        out_image = np.zeros((2*44, 2*34))

        image_features0 = self.feature_extractor(image[:, 0:320, 0:240, :])
        image_features1 = self.feature_extractor(image[:, 0:320, 240:, :])
        image_features2 = self.feature_extractor(image[:, 320:, 0:240, :])
        image_features3 = self.feature_extractor(image[:, 320:, 240:, :])

        #from: ('b', '0', '1', 'c')
        # to : ('b', 'c', '0', '1')
        image_features0 = np.rollaxis(image_features0, 1, 4)
        image_features1 = np.rollaxis(image_features1, 1, 4)
        image_features2 = np.rollaxis(image_features2, 1, 4)
        image_features3 = np.rollaxis(image_features3, 1, 4)

        classified_image0 = self.classifier(image_features0)
        classified_image1 = self.classifier(image_features1)
        classified_image2 = self.classifier(image_features2)
        classified_image3 = self.classifier(image_features3)

        out_image[0:44, 0:34] = classified_image0
        out_image[0:44, 34:] = classified_image1
        out_image[44:, 0:34] = classified_image2
        out_image[44:, 34:] = classified_image3

        return out_image


def get_test_image(image_id=0):
    hdf5_dataset_filename = PYLEARN_DATA_PATH + "/nyu_depth_labeled/rgbd_preprocessed.h5"
    dataset = h5py.File(hdf5_dataset_filename)
    rgbd_image = dataset['rgbd'][image_id][:]
    return np.expand_dims(rgbd_image, 0)


#returns the string label for a given integer between 0-894
def get_pixel_label(dataset, output=755):
    pixel_label = ""
    for i in dataset[dataset['names'][0][output]][:]:
        pixel_label += chr(i)
    return pixel_label


if __name__ == "__main__":

    classifier = DenseRGBDClassifier(CONV_MODEL_FILENAME)

    for i in range(1447):
        print 'looking at image:' + str(i)
        image = get_test_image(i)

        classified_image = classifier.run(image)

        counter = collections.Counter(classified_image.flatten())

        figure_text = ""
        for key in counter.keys():
            text_label = get_pixel_label(NYU_DEPTH_V2_DATASET, key)
            label_frequency = counter[key]
            figure_text += str(text_label) + ": " + str(label_frequency) + "\n"

        figure = pylab.figure()
        figure.text(.1, .1, figure_text)

        figure.add_subplot(2, 1, 0)
        pylab.imshow(image[0, :, :, 0:3])

        figure.add_subplot(2, 1, 1)
        pylab.imshow(classified_image)

        pylab.show()

        #segmented_image = slic_image.get_slic_image_segments(image)

        #import IPython
        #IPython.embed()





