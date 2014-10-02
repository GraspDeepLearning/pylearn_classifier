

import os
import h5py
import time

from grasp_classification_pipeline import GraspClassificationPipeline

import paths

CROP_BORDER_DIM = 15


#Lets the user choose an existing pylearn model to run
def get_model():

    models = os.listdir(paths.MODEL_DIR)

    print
    print "Choose model: "
    print

    for i in range(len(models)):
        print str(i) + ": " + models[i]

    print
    model_index = int(raw_input("Enter Id of model to run (ex 0, 1, or 2): "))

    model = models[model_index]

    return model


#Allows the user to choose a specific dataset to run the model over
def get_dataset_file():

    datasets = os.listdir(paths.RAW_TRAINING_DATASET_DIR)

    print
    print "Choose dataset file: "
    print

    for i in range(len(datasets)):
        print str(i) + ": " + datasets[i]

    print
    dataset_index = int(raw_input("Enter Id of dataset file (ex 0, 1, or 2): "))
    dataset_file = datasets[dataset_index]

    return dataset_file


def get_save_path(input_data_file, input_model_file):
    return paths.HEATMAPS_DATASET_DIR + input_data_file[:-3] + '_' + input_model_file + '.h5'




def save(rgbd_img, heatmaps, image_index, save_file, num_images):
    if 'rgbd_data' not in save_file.keys():
        save_file.create_dataset('rgbd_data', (num_images, rgbd_img.shape[0], rgbd_img.shape[1], rgbd_img.shape[2]),
                                 chunks=(10, rgbd_img.shape[0], rgbd_img.shape[1], rgbd_img.shape[2]))

    if 'heatmaps' not in save_file.keys():
        save_file.create_dataset('heatmaps', (num_images, heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2]),
                                 chunks=(10, heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2]))

    save_file['rgbd_data'][image_index] = rgbd_img
    save_file['heatmaps'][image_index] = heatmaps


if __name__ == "__main__":

    conv_model_name = get_model()
    conv_model_filepath = paths.MODEL_DIR + conv_model_name + "/cnn_model.pkl"

    dataset_file = get_dataset_file()
    dataset_filepath = paths.RAW_TRAINING_DATASET_DIR + dataset_file

    save_filepath = get_save_path(dataset_file, conv_model_name)
    save_file = h5py.File(save_filepath)

    grasp_classification_pipeline = GraspClassificationPipeline(conv_model_filepath, border_dim=CROP_BORDER_DIM, useFloat64=False)

    dataset = h5py.File(dataset_filepath)

    rgbd_images = dataset['rgbd_data']

    num_images = rgbd_images.shape[0]

    for image_index in range(num_images):

        print str(image_index) + "/" + str(num_images)

        rgbd_img = rgbd_images[image_index]

        start = time.time()
        heatmaps = grasp_classification_pipeline.run(rgbd_img)
        print time.time() - start

        save(rgbd_img, heatmaps, image_index, save_file, num_images)



