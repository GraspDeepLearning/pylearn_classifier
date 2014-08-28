import cPickle
import os.path


def get_model(model_path, over_write=False):
    if os.path.isfile(model_path) and not over_write:
        return cPickle.load(model_path)
    else:
        pass


