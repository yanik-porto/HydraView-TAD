from .dataset_mv import DatasetMV

import copy
import numpy as np

pkummd_cam_map = {'L': 0, 'M': 1, 'R': 2}
class DatasetWindowTAD(DatasetMV):
    def __init__(self, data_path, dataset_name, split='xsub_train', n_views=2, num_classes=120, preprocessing=False, classes_map=None, label_map=None, window_size=16, **kwargs):
        super().__init__(data_path, dataset_name, split, n_views, num_classes, classes_map, preprocessing, label_map, **kwargs)

        for idx_ok in self.map_idx_ok:
            self.data[idx_ok] = self.split_data_by_window(self.data[idx_ok], window_size)

    def split_data_by_window(self, data_item, window_size):
        kpts = data_item['keypoint'] if 'keypoint' in data_item else data_item['joints']  # shape (num_person, num_frames, num_joints, 2 or 3)
        if len(kpts.shape) < 4:
            kpts = np.expand_dims(kpts, axis=0) # shape (num_person, num_frames, num_joints, 2 or 3)
        num_frames = kpts.shape[1]
        assert 'binary_labels' in data_item, "DatasetWindowClassif requires binary_labels in data_item : data item keys are " + str(data_item.keys())
        binary_labels = data_item['binary_labels'] # shape (num_peron, num_frames, num_classes)
        binary_labels = binary_labels[0] if len(binary_labels.shape) == 3 else binary_labels  # shape (num_frames, num_classes)
        assert num_frames == len(binary_labels), f"Number of frames {num_frames} does not match number of binary labels {len(binary_labels)}"

        kpts_windows = []
        binary_labels_windows = []
        for start_idx in range(0, num_frames, window_size):
            end_idx = min(num_frames, start_idx + window_size)
            window_kpts = np.zeros((kpts.shape[0], window_size, kpts.shape[2], kpts.shape[3]), dtype=kpts.dtype)
            window_kpts[:, :end_idx - start_idx] = kpts[:, start_idx:end_idx] # shape (num_person, window_size, num_joints, 2 or 3)
            kpts_windows.append(window_kpts)

            binary_labels_window = binary_labels[start_idx:end_idx]
            num_classes = binary_labels_window.shape[-1]
            # for TAD, we keep one set of binary labels for the entire window
            new_binary_labels = np.zeros(num_classes, dtype=np.float32) # shape (num_classes)
            unique_labels = np.unique(np.where(binary_labels_window == 1)[1]) 
            for ul in unique_labels:
                new_binary_labels[ul] = 1.0
            binary_labels_windows.append(new_binary_labels)

        assert len(kpts_windows) == len(binary_labels_windows), f"{len(kpts_windows)} vs {len(binary_labels_windows)}"
        # print(f"Split data item with {num_frames} frames into {len(kpts_windows)} windows of size {window_size}")
        # print(f"Original binary labels shape: {binary_labels.shape}, windowed binary labels shape: {np.array(binary_labels_windows).shape}")
        kpts_windows = np.stack(kpts_windows, axis=0)  # shape (num_windows, num_person, window_size, num_joints, 2 or 3)
        binary_labels_windows = np.array(binary_labels_windows)  # shape (num_windows, num_classes)
        data_item["keypoint" if 'keypoint' in data_item else 'joints'] = kpts_windows
        data_item['total_frames'] = window_kpts.shape[1]
        data_item['binary_labels'] = binary_labels_windows
        return data_item
    
    def __getitem__(self, idx_required):
        """Get the sample for either training or testing given index."""

        assert(len(self.map_idx_ok) > idx_required)

        idx_mapped = self.map_idx_ok[idx_required]
        
        d = copy.deepcopy(self.data[idx_mapped])

        samples = [dict()] * self.n_views
        samples[0] = self.preprocessing(d)

        if self.n_views > 1:
            _, camCurrent = self.get_group_from_name(d["frame_dir"])
            for id, (camAssoc, idx_assoc) in enumerate(d["assocs"]):
                if self.respect_cams_order:
                    mapAssocs = self.map_assocs_samples[camCurrent]
                    if camAssoc not in mapAssocs:
                        continue
                    idxInSamples = mapAssocs.index(camAssoc) + 1 # + 1 for current camera
                else:
                    idxInSamples = id + 1
                if idxInSamples >= self.n_views:
                    continue

                ass = copy.deepcopy(self.data[idx_assoc])
                samples[idxInSamples] = self.preprocessing(ass)

        for s in samples:
            assert len(s.keys()) > 0, str(camCurrent) + " : " + str(d["assocs"])

        return self.merge_samples(samples)