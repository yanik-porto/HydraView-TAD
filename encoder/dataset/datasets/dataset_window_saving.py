import sys
import os
import numpy as np
from torch.utils.data import Dataset
from itertools import chain

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from preprocessors import *

class DatasetWindowForSaving(Dataset):
    def __init__(self, data_path, preprocessing, window_size=16, label_map=None, filter_with_labels=False):
        super().__init__()
        assert os.path.exists(data_path), data_path + " not found"

        self.preprocessing = preprocessing

        self.label_map = [x.strip() for x in open(op.join(op.dirname(__file__), label_map)).readlines()] if label_map is not None else []

        data_frames = np.load(data_path, allow_pickle=True)
        if "kpts" in data_frames:
            data_frames = data_frames["kpts"] # shape (num_frames, num_joints, 2)

        if len(data_frames.shape) < 4:
            data_frames = np.expand_dims(data_frames, axis=0) # shape (num_person, num_frames, num_joints, 2)

        if filter_with_labels:
            assert self.label_map is not None
            seq_name = os.path.basename(os.path.splitext(data_path)[0])
            labels_path = os.path.join(op.dirname(data_path), seq_name + "_labels.pkl")
            if not os.path.exists(labels_path):
                raise FileNotFoundError(labels_path + " not found")
            labels = pickle.load(open(labels_path, 'rb'))
            labels = list(labels.values())
            assert len(labels) == data_frames.shape[1]

        atLeastOne = False
        self.data_windows = [] # list of windows of shape (num_person, window_size, num_joints, 2)
        num_frames = data_frames.shape[1]
        for start_idx in range(0, num_frames, window_size):
            end_idx = min(num_frames, start_idx + window_size)
            window = data_frames[:, start_idx:end_idx]  # shape (num_peron, window_size, num_joints, 2)
            self.data_windows.append(window)

            if filter_with_labels:
                labels_window = labels[start_idx:end_idx]
                labels_window = list(chain.from_iterable([list(f) for f in labels_window]))
                label_set = list(set(labels_window) & set(self.label_map))
                if len(label_set) > 0:
                    atLeastOne = True

        if filter_with_labels and not atLeastOne:
            self.data_windows = []
            print("Warning: No corresponding label found for this sequence :", data_path)

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.data_windows)

    def __getitem__(self, idx_required):

        data = self.data_windows[idx_required]

        if len(data.shape) == 3: # case of single person in the frame
            data = np.expand_dims(data, axis=0)

        data_dict = {
            'total_frames': data.shape[1]
        }
        if data.shape[-1] == 2:
            data_dict['keypoint'] = data.astype(np.float32)  # shape (num_persons, window_size, num_joints, 2)
        elif data.shape[-1] == 3:
            data_dict['joints'] = data.astype(np.float32)  # shape (num_persons, window_size, num_joints, 3)


        if self.preprocessing:
            data_dict = self.preprocessing(data_dict)

        return data_dict
  