
from grasp_classification_pipeline import *

import paths
import os

def init_save_file(input_data_file, input_model_file):

    dataset_filepath = paths.HEATMAPS_DATASET_DIR + input_data_file[:-3] + '_' + input_model_file + '4.h5'

    if os.path.exists(dataset_filepath):
        os.remove(dataset_filepath)

    h5py.File(dataset_filepath)

    return dataset_filepath


def main():

    conv_model_name = paths.choose_from(paths.MODEL_DIR)
    conv_model_filepath = paths.MODEL_DIR + conv_model_name + "/cnn_model.pkl"

    dataset_file = paths.choose_from(paths.RAW_TRAINING_DATASET_DIR)
    raw_rgbd_filepath = paths.RAW_TRAINING_DATASET_DIR + dataset_file

    #priors_filepath = paths.PRIORS_DIR + 'saxena_rect_priors.h5'

    save_filepath = init_save_file(dataset_file, conv_model_name)

    pipeline = GraspClassificationPipeline(save_filepath, raw_rgbd_filepath)

    pipeline.add_stage(CopyInRaw(raw_rgbd_filepath, in_key='image', out_key='rgbd_data'))
    pipeline.add_stage(LecunSubtractiveDivisiveLCN(in_key='rgbd_data', out_key='rgbd_data_normalized'))
    pipeline.add_stage(FeatureExtraction(conv_model_filepath, use_float_64=False))
    pipeline.add_stage(Classification(conv_model_filepath))
    pipeline.add_stage(HeatmapNormalization())
    #pipeline.add_stage(ConvolvePriors(priors_filepath))
    #pipeline.add_stage(CalculateMax())
    #pipeline.add_stage(CalculateTopFive(input_key='convolved_heatmaps', output_key='dependent_grasp_points'))
    #pipeline.add_stage(CalculateTopFive(input_key='normalized_heatmaps', output_key='independent_grasp_points'))

    pipeline.run()

if __name__ == "__main__":
    main()
