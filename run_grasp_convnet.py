
from grasp_classification_pipeline import *

import paths
import os


# DATA_SIZES = dict(rgbd_data=(900, 480, 640, 4),
#                   rgbd_data_normalized=(900, 480, 640, 4),
#                   extracted_features=(900, 52, 72, 64),
#                   heatmaps=(900, 52, 72, 3),
#                   normalized_heatmaps=(900, 52, 72, 3),
#                   convolved_heatmaps=(900,  317, 477, 3),
#                   best_grasp=(900, 317, 477, 3),
#                   l_convolved_heatmaps=(900, 6, 317, 477),
#                   p_convolved_heatmaps=(900, 6, 317, 477),
#                   r_convolved_heatmaps=(900, 6,  317, 477),
#                   independent_grasp_points=(900, 3, 480, 640, 3),
#                   dependent_grasp_points=(900, 3, 480, 640, 3)
#                   )
#
# CHUNK_SIZES = dict(rgbd_data=(10, 480, 640, 4),
#                    rgbd_data_normalized=(10, 480, 640, 4),
#                    extracted_features=(10, 52, 72, 64),
#                    heatmaps=(10, 52, 72, 3),
#                    normalized_heatmaps=(10, 52, 72, 3),
#                    convolved_heatmaps=(10,  317, 477, 3),
#                    best_grasp=(10,  317, 477, 3),
#                    l_convolved_heatmaps=(10, 6,  317, 477),
#                    p_convolved_heatmaps=(10, 6, 317, 477),
#                    r_convolved_heatmaps=(10, 6,  317, 477),
#                    independent_grasp_points=(10, 3, 480, 640, 3),
#                    dependent_grasp_points=(10, 3, 480, 640, 3)
#                    )

DATA_SIZES = dict(rgbd_data=(900, 480, 640, 4),
                  rgbd_data_normalized=(900, 480, 640, 4),
                  extracted_features=(900, 11, 16, 256),
                  heatmaps=(900, 11, 16, 3),
                  normalized_heatmaps=(900, 11, 16, 3),
                  convolved_heatmaps=(900,  317, 477, 3),
                  best_grasp=(900, 317, 477, 3),
                  l_convolved_heatmaps=(900, 6, 317, 477),
                  p_convolved_heatmaps=(900, 6, 317, 477),
                  r_convolved_heatmaps=(900, 6,  317, 477),
                  independent_grasp_points=(900, 3, 480, 640, 3),
                  dependent_grasp_points=(900, 3, 480, 640, 3)
                  )

CHUNK_SIZES = dict(rgbd_data=(10,480, 640, 4),
                   rgbd_data_normalized=(10, 480, 640, 4),
                   extracted_features=(10, 11, 16, 64),
                   heatmaps=(10, 11, 16,  3),
                   normalized_heatmaps=(10, 11, 16, 3),
                   convolved_heatmaps=(10,  317, 477, 3),
                   best_grasp=(10,  317, 477, 3),
                   l_convolved_heatmaps=(10, 6,  317, 477),
                   p_convolved_heatmaps=(10, 6, 317, 477),
                   r_convolved_heatmaps=(10, 6,  317, 477),
                   independent_grasp_points=(10, 3, 480, 640, 3),
                   dependent_grasp_points=(10, 3, 480, 640, 3)
                   )

#For YLI
# DATA_SIZES = dict(rgbd_data=(900, 1024, 1280, 1),
#                   rgbd_data_normalized=(900,1024, 1280, 1),
#                   extracted_features=(900, 28, 36, 256),
#                   heatmaps=(900, 28, 36, 3),
#                   normalized_heatmaps=(900, 28, 36, 3),
#                   convolved_heatmaps=(900,  317, 477, 3),
#                   best_grasp=(900, 317, 477, 3),
#                   l_convolved_heatmaps=(900, 6, 317, 477),
#                   p_convolved_heatmaps=(900, 6, 317, 477),
#                   r_convolved_heatmaps=(900, 6,  317, 477),
#                   independent_grasp_points=(900, 3, 480, 640, 3),
#                   dependent_grasp_points=(900, 3, 480, 640, 3)
#                   )
#
# CHUNK_SIZES = dict(rgbd_data=(10,1024, 1280, 1),
#                    rgbd_data_normalized=(10, 1024, 1280, 1),
#                    extracted_features=(10, 28, 36, 64),
#                    heatmaps=(10, 28, 36, 3),
#                    normalized_heatmaps=(10, 28, 36, 3),
#                    convolved_heatmaps=(10,  317, 477, 3),
#                    best_grasp=(10,  317, 477, 3),
#                    l_convolved_heatmaps=(10, 6,  317, 477),
#                    p_convolved_heatmaps=(10, 6, 317, 477),
#                    r_convolved_heatmaps=(10, 6,  317, 477),
#                    independent_grasp_points=(10, 3, 480, 640, 3),
#                    dependent_grasp_points=(10, 3, 480, 640, 3)
#                    )


def init_save_file(input_data_file, input_model_file):

    dataset_filepath = paths.HEATMAPS_DATASET_DIR + input_data_file[:-3] + '_' + input_model_file + '4.h5'

    if os.path.exists(dataset_filepath):
        os.remove(dataset_filepath)

    dataset = h5py.File(dataset_filepath)

    print
    print "Initing Dataset with: "
    print

    for key in DATA_SIZES.keys():
        print str(key) + " shape=" + str(DATA_SIZES[key]) + " chunks=" + str(CHUNK_SIZES[key])
        dataset.create_dataset(key, DATA_SIZES[key], chunks=CHUNK_SIZES[key])

    print

    return dataset_filepath


def main():
    conv_model_name = paths.choose_from(paths.MODEL_DIR)
    conv_model_filepath = paths.MODEL_DIR + conv_model_name + "/cnn_model.pkl"

    dataset_file = paths.choose_from(paths.RAW_TRAINING_DATASET_DIR)
    raw_rgbd_filepath = paths.RAW_TRAINING_DATASET_DIR + dataset_file

    priors_filepath = paths.PRIORS_DIR + 'saxena_rect_priors.h5'

    save_filepath = init_save_file(dataset_file, conv_model_name)

    pipeline = GraspClassificationPipeline(save_filepath, raw_rgbd_filepath)

    pipeline.add_stage(CopyInRaw(raw_rgbd_filepath))
    pipeline.add_stage(NormalizeRaw())
    pipeline.add_stage(SlidingWindowNormalization(window_size=(170, 170), key="rgbd_data"))
    pipeline.add_stage(FeatureExtraction(conv_model_filepath, useFloat64=False))
    pipeline.add_stage(Classification(conv_model_filepath))
    pipeline.add_stage(Normalization())
    pipeline.add_stage(ConvolvePriors(priors_filepath))
    pipeline.add_stage(CalculateMax())
    #pipeline.add_stage(CalculateTopFive(input_key='convolved_heatmaps', output_key='dependent_grasp_points'))
    #pipeline.add_stage(CalculateTopFive(input_key='normalized_heatmaps', output_key='independent_grasp_points'))

    pipeline.run()

if __name__ == "__main__":
    main()
