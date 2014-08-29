
import os
from pylearn2.testing import skip
from pylearn2.config import yaml_parse

def get_grasp_72_72_model():

    yaml_path = os.path.abspath(os.path.dirname(__file__)) + "/models/grasp_72x72_model"
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/models/grasp_72x72_model"

    yaml = open("{0}/model.yaml".format(yaml_path), 'r').read()
    hyper_params = {'batch_size': 25,
                    'output_channels_h0': 32,
                    'output_channels_h1': 64,
                    'output_channels_h2': 64,
                    'max_epochs': 5000,
                    'save_path': save_path}

    yaml_with_hyper_params = yaml % hyper_params
    return yaml_with_hyper_params

def get_grasp_72_72_maxout_model():

    yaml_path = os.path.abspath(os.path.dirname(__file__)) + "/models/grasp_72x72_maxout_model"
    save_path = os.path.dirname(os.path.realpath(__file__)) + "/models/grasp_72x72_maxout_model"

    yaml = open("{0}/maxout_model.yaml".format(yaml_path), 'r').read()
    hyper_params = {'batch_size': 10,
                    'output_channels_h0': 32,
                    'output_channels_h1': 32,
                    'output_channels_h2': 32,
                    'max_epochs': 5000,
                    'save_path': save_path}

    yaml_with_hyper_params = yaml % hyper_params
    return yaml_with_hyper_params

def train_convolutional_network(yaml_with_hyper_params):

    skip.skip_if_no_data()

    train = yaml_parse.load(yaml_with_hyper_params)

    train.main_loop()


if __name__ == "__main__":
    model_yaml = get_grasp_72_72_model()

    train_convolutional_network(model_yaml)
