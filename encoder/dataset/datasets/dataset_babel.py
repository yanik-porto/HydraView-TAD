from .dataset_babel_mv import DatasetBabelMV

import sys
import os.path as op
sys.path.insert(0, op.join(op.dirname(__file__), '../'))

class DatasetBabel(DatasetBabelMV):
    def __init__(self, data_path, split='xsub_train', n_views=2, num_classes=120, preprocessing=False, classes_map=None, label_map=None, oneshot=False, motionid_labels_path="", **kwargs):
        self.motionid_to_labels = self.load_motionid_to_labels(motionid_labels_path, split)
        super().__init__(data_path, "babel", split, n_views, num_classes, preprocessing, classes_map, label_map, oneshot, **kwargs)
