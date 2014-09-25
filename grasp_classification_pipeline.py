
import theano
import pylearn2
import cPickle

import numpy as np


class Classification():

    def __init__(self, model_filepath):

        f = open(model_filepath)

        cnn_model = cPickle.load(f)
        cnn_model = cnn_model.layers[-1]

        W = cnn_model.get_weights_topo()
        W = W[0, 0, :, :]

        b = cnn_model.b.get_value()

        self.W = W
        self.b = b

    def run(self, X):
        return np.dot(X, self.W) + self.b


class FeatureExtraction():

    def __init__(self, model_filepath):

        f = open(model_filepath)
        cnn_model = cPickle.load(f)

        new_space = pylearn2.space.Conv2DSpace((480, 640), num_channels=4, axes=('c', 0, 1, 'b'), dtype='float32')

        cnn_model.layers = cnn_model.layers[0:-1]

        #we want to padd zeros around the edges rather than ignoring edge pixels
        for i in range(len(cnn_model.layers)):
            cnn_model.layers[i].border_mode = "full"

        cnn_model.set_input_space(new_space)

        X = cnn_model.get_input_space().make_theano_batch()
        Y = cnn_model.fprop(X)

        self._feature_extractor = theano.function([X], Y)

    def run(self, img_in):
        img = np.zeros((4, 480, 640, 1), dtype=np.float32)
        img[:, :, :, 0] = np.rollaxis(img_in, 2, 0)
        out_raw = self._feature_extractor(img)
        out_rolled = np.rollaxis(out_raw, 1, 4)
        out_window = out_rolled[0, :, :, :]

        return out_window


class Normalization():

    def run(self, output):
        #currently output.min() is < 0 and output.max() > 0
        #normalized between 0 and 255
        return 255 - (output-output.min())/(output.max()-output.min())*255.0


class Crop():

    def __init__(self, border_dim=15):
        self.border_dim = border_dim

    def run(self, output):
        return output[self.border_dim:-self.border_dim, self.border_dim:-self.border_dim]




class GraspClassificationPipeline():

    def __init__(self, model_filepath):

        self.model_filepath = model_filepath
        self._feature_extraction = FeatureExtraction(model_filepath)
        self._classification = Classification(model_filepath)
        self._normalization = Normalization()
        self._crop = Crop()

    def run(self, img_in, normalize=True, crop_border=True):

        features = self._feature_extraction.run(img_in)
        output = self._classification.run(features)

        if normalize:
            output = self._normalization.run(output)

        if crop_border:
            output = self._crop.run(output)

        return output