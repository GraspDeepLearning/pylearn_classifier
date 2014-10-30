
import paths
import choose
import matplotlib.pyplot as plt
import h5py

dataset_file = choose.choose_from(paths.PROCESSED_TRAINING_DATASET_DIR)
processed_rgbd_filepath = paths.PROCESSED_TRAINING_DATASET_DIR + dataset_file

dataset = h5py.File(processed_rgbd_filepath)

print dataset.keys()

train_patches = dataset['train_patches']
train_patch_labels = dataset['train_patch_labels']

i = 0
while True:
    plt.figure(0)

    for index in range(9):
        plt.subplot(3, 3, index + 1)
        plt.imshow(train_patches[i+index, :, :, 0:3])
        plt.title(str(train_patch_labels[i+index]))

    i += 9
    plt.show()
