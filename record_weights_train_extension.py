from pylearn2.train_extensions import TrainExtension
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math


class RecordWeights(TrainExtension):

    def __init__(self, output_dir_path):
        self.outout_dir_path = output_dir_path
        self.current_weight_file_number = 0

    def on_monitor(self, model, dataset, algorithm):

        #get the current weights from the model
        weights = model.get_weights_topo()

        num_weights = weights.shape[0]

        #set up the dimensions of the figure
        w = math.ceil(math.sqrt(num_weights))
        h = w

        plt.figure()
        for i in range(num_weights):
            # add a new subplot for each kernel
            sub = plt.subplot(h, w, i + 1)
            #remove the axis so it looks prettier
            sub.axes.get_xaxis().set_visible(False)
            sub.axes.get_yaxis().set_visible(False)
            #make it greyscale
            plt.imshow(weights[i, :, :, 0], cmap=cm.Greys_r)

        plt.savefig(self.outout_dir_path + '/weight_' + str(self.current_weight_file_number) + '.png')

        #must close this or else plt leaks
        plt.close()

        self.current_weight_file_number += 1