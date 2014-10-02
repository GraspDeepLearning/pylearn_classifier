
import os
import time

from pylearn2.testing import skip
from pylearn2.config import yaml_parse

import paths


#helper to build up the save path from the name of the model and the current time.
def get_save_path(model_template, dataset):
    t = time.localtime()

    minute = str(t.tm_min)
    if len(minute) == 1:
        minute = '0' + minute

    t_string = str(t.tm_mon) + "_" + str(t.tm_mday) + "_" + str(t.tm_hour) + "_" + minute

    #remove '.h5' from the dataset
    dataset = dataset[:-3]

    return paths.MODEL_DIR + dataset + "_" + model_template + "_" + t_string


#the model template is the model without the save path and dataset specified
def get_model_template():

    model_templates = os.listdir(paths.MODEL_TEMPLATE_DIR)

    print
    print "Choose model: "
    print

    for i in range(len(model_templates)):
        print str(i) + ": " + model_templates[i]

    print
    model_template_index = int(raw_input("Enter Id of model to train (ex 0, 1, or 2): "))

    model_template = model_templates[model_template_index]

    return model_template


#the dataset we are going to train the model against
def get_dataset():

    datasets = os.listdir(paths.PROCESSED_TRAINING_DATASET_DIR)

    print
    print "Choose dataset: "
    print

    for i in range(len(datasets)):
        print str(i) + ": " + datasets[i]

    print
    dataset_index = int(raw_input("Enter Id of dataset to train on (ex 0, 1, or 2): "))

    dataset = datasets[dataset_index]

    return dataset


#have the user choose:
# 1) a model template
# 2) a dataset to train against.
#then modify the hyper_params to specify a save location as well as the dataset used.
def build_model():

    model_template = get_model_template()
    dataset = get_dataset()

    model_template_yaml = open(paths.MODEL_TEMPLATE_DIR + model_template + "/model.yaml", 'r').read()
    hyper_params_file = open(paths.MODEL_TEMPLATE_DIR + model_template + "/hyper_params.yaml", 'r').read()

    hyper_params_dict = yaml_parse.load(hyper_params_file)
    hyper_params_dict['save_path'] = get_save_path(model_template, dataset)
    hyper_params_dict['dataset'] = paths.RAW_TRAINING_DATASET_DIR + dataset

    return model_template_yaml, hyper_params_dict


#we want to remove the directory for the save model if it exists, then we want to
#save the hyper params and model_yaml that were used.  This way, we have a copy of them
#even if the template files are changed.
def prep_model_save_path(save_path, model_yaml, hyper_params_dict):

    assert paths.MODEL_DIR in save_path

    if os.path.exists(save_path):
        for file_name in os.listdir(save_path):
            os.remove(save_path + '/' + file_name)
        os.rmdir(save_path)

    os.mkdir(save_path)

    f = open(save_path + '/' + 'model.yaml', 'w')
    f.write(model_yaml)
    f.close()

    f = open(save_path + '/' + 'hyper_params.yaml', 'w')
    for key in hyper_params_dict.keys():
        f.write(str(key) + ': ' + str(hyper_params_dict[key]) + '\n')
    f.close()


#this method verifies that we have data, and actually begins training the model
def train_convolutional_network(model_with_hyper_params):
    skip.skip_if_no_data()
    train = yaml_parse.load(model_with_hyper_params)
    train.main_loop()


def main():
    #ask user for input in order to get the model and hyper parameters we want to use
    model_template_yaml, hyper_params_dict = build_model()

    #save the model and hyper params in the save dir so that we can access them
    #whenever we want, even if the template files have been changed.
    prep_model_save_path(hyper_params_dict['save_path'], model_template_yaml, hyper_params_dict)

    #insert the hyper parameters into the model template
    model = model_template_yaml % hyper_params_dict

    #train the model.
    train_convolutional_network(model)

if __name__ == "__main__":
    main()


