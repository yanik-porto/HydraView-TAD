from .dataset_mv import DatasetMV
from .dataset_babel_mv import DatasetBabelMV
from .dataset_mv_tad import DatasetMVTAD
from .dataset_window_saving import DatasetWindowForSaving
from .dataset_window_classif import DatasetWindowClassif
from .dataset_window_tad import DatasetWindowTAD
from preprocessors import *

import os.path as op

def create_dataset(config, setname="train12"):
    assert setname in config, "no " + setname + " specified in config"
    assert 'dataset' in config[setname], "no dataset specified in config"

    preprocessing = create_preprocessing(config[setname])

    cfg_dataset = config[setname]['dataset']

    name = cfg_dataset.get('name', None)

    dataset = None
    if name in ("babel_mv", "babel", "pkummd"):
        if name == "babel_mv":
            dataset = DatasetBabelMV(op.join(op.dirname(__file__), cfg_dataset['file']), split=cfg_dataset["split"], preprocessing=preprocessing, **cfg_dataset["params"])
        else:
            dataset = DatasetMV(op.join(op.dirname(__file__), cfg_dataset['file']), name, split=cfg_dataset["split"], preprocessing=preprocessing, **cfg_dataset["params"])
    elif name in ("babel_tad","pkummd_tad"):
        dataset = DatasetMVTAD(op.join(op.dirname(__file__), cfg_dataset['file']), name, split=cfg_dataset["split"], preprocessing=preprocessing, **cfg_dataset["params"])
    elif name == 'window_for_saving':
        dataset = DatasetWindowForSaving(op.join(op.dirname(__file__), cfg_dataset['file']), preprocessing=preprocessing, **cfg_dataset["params"])
    elif name == 'window_classif':
        dataset = DatasetWindowClassif(op.join(op.dirname(__file__), cfg_dataset['file']), name, split=cfg_dataset["split"], preprocessing=preprocessing, **cfg_dataset["params"])
    elif name == 'window_tad':
        dataset = DatasetWindowTAD(op.join(op.dirname(__file__), cfg_dataset['file']), name, split=cfg_dataset["split"], preprocessing=preprocessing, **cfg_dataset["params"])
    else:
        print(name, " not handled yet in datasets")

    return dataset