from .dataset_mv import DatasetMV

import copy
import numpy as np

class DatasetWindowClassif(DatasetMV):
    """
    Split data into windows for classification
    """
    def __init__(self, data_path, dataset_name, split='xsub_train', n_views=2, num_classes=120, preprocessing=False, classes_map=None, label_map=None, window_size=16, **kwargs):
        super().__init__(data_path, dataset_name, split, n_views, num_classes, classes_map, preprocessing, label_map, **kwargs)

        data_by_window = []
        for idx_ok in self.map_idx_ok:
            data_item = self.data[idx_ok]
            data_item_windows = self.split_data_by_window(data_item, window_size)
            data_by_window.extend(data_item_windows)

        self.data = data_by_window
        self.map_idx_ok = [idx for idx, item in enumerate(self.data) if 'keypoint' or 'joints' in item]

    def split_data_by_window(self, data_item, window_size):
        kpts = data_item['keypoint'] if 'keypoint' in data_item else data_item['joints']  # shape (num_person, num_frames, num_joints, 2 or 3)
        num_frames = kpts.shape[1]
        assert 'binary_labels' in data_item, "DatasetWindowClassif requires binary_labels in data_item : data item keys are " + str(data_item.keys())
        binary_labels = data_item['binary_labels'] # shape (num_peron, num_frames, num_classes)
        binary_labels = binary_labels[0] if len(binary_labels.shape) == 3 else binary_labels  # shape (num_frames, num_classes)
        assert num_frames == len(binary_labels), f"Number of frames {num_frames} does not match number of binary labels {len(binary_labels)}: {kpts.shape} vs {binary_labels.shape}"

        data_windows = []
        for start_idx in range(0, num_frames, window_size):
            end_idx = min(num_frames, start_idx + window_size)
            window_kpts = kpts[:, start_idx:end_idx]  # shape (num_person, window_size, num_joints, 2 or 3)

            if len(window_kpts.shape) < 4:
                window_kpts = np.expand_dims(window_kpts, axis=0) # shape (num_person, num_frames, num_joints, 2 or 3)

            data_window = copy.deepcopy(data_item)
            data_window['keypoint' if 'keypoint' in data_item else 'joints'] = window_kpts
            data_window['total_frames'] = window_kpts.shape[1]

            binary_labels_window = binary_labels[start_idx:end_idx]
            data_window['binary_labels'] = binary_labels_window
            # For classification, we collect all labels present in the window and create as many samples as there are labels
            unique_labels = np.unique(np.where(binary_labels_window == 1)[1]) # TODO : check if last index which corresponds to no action should be removed
            for ul in unique_labels:
                data_window_copy = copy.deepcopy(data_window)
                data_window_copy['label'] = int(ul)
                data_windows.append(data_window_copy)
        return data_windows