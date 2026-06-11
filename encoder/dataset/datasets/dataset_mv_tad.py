from .dataset_mv import DatasetMV

import copy
import numpy as np

pkummd_cam_map = {'L': 0, 'M': 1, 'R': 2}
class DatasetMVTAD(DatasetMV):
    """
    The dataset MVTAD load pre-computed features to train models for TAD in single and multiview systems.
    """
    def __init__(self, data_path, dataset_name, split='xsub_train', n_views=2, num_classes=120, preprocessing=False, classes_map=None, label_map=None, sorted_by_pair=True, **kwargs):
        self.sorted_by_pair = sorted_by_pair
        super().__init__(data_path, dataset_name, split, n_views, num_classes, classes_map, preprocessing, label_map, **kwargs)


    def save_binarylabels(self, data):
        pass

    def find_associated_ids(self, data):
        map_index_with_assocs = []

        if self.n_views == 1:
            map_index_with_assocs = list(range(len(data)))
            if self.text_embeds is not None:
                 for d in data: d["text_embed"] = self.text_embeds[d["label"]]

            return data, map_index_with_assocs

        if self.sorted_by_pair:
            for idx in range(0, len(data), 2):
                groupCurrent, camCurrent = self.get_group_from_name(data[idx]["frame_dir"])
                groupOther, camOther = self.get_group_from_name(data[idx+1]['frame_dir'])
                assert groupCurrent == groupOther, f"Data not sorted by group ? {data[idx]['frame_dir']} vs {data[idx+1]['frame_dir']}"
                assert camCurrent != camOther, f"Same camera for two consecutive samples ? {data[idx]['frame_dir']} vs {data[idx+1]['frame_dir']}"
                assocs = [(camOther, idx+1)]

                if len(assocs) + 1 < self.n_views:
                    print("index #", idx, " (with  name ", data[idx]["frame_dir"], " belongs to a group with too few views : ", len(assocs) + 1, " < ", self.n_views)
                    continue
                
                data[idx]["assocs"] = assocs

                if self.text_embeds is not None:
                    data[idx]["text_embed"] = self.text_embeds[data[idx]["label"]]

                map_index_with_assocs.append(idx)
        else:

            return super().find_associated_ids(data)

        return data, map_index_with_assocs
    
    def merge_annot_into_other(self, annot, other, fit_annot_length=False):
        concat = copy.copy(other)

        assert 'keypoint' in concat, str(concat.keys())
        assert 'keypoint' in annot, str(annot.keys())


        if not fit_annot_length:
            T = max(concat['keypoint'].shape[0], annot['keypoint'].shape[0])
            concat['total_frames'] = T

            os = annot['keypoint'].shape
            other_kpts = np.zeros((T, *os[1:]), np.float32)
            other_kpts[:annot['keypoint'].shape[0], :, :, :] = annot['keypoint']

            concat['keypoint'] = np.concatenate((concat['keypoint'], other_kpts), axis=2) # [T, 1, M, C]
        
        else:
            # Fit annot length to other length
            T_other = other['keypoint'].shape[0]
            concat['total_frames'] = T_other

            os = annot['keypoint'].shape
            annot_kpts = np.zeros((T_other, *os[1:]), np.float32)
            if annot['keypoint'].shape[0] >= T_other:
                annot_kpts = annot['keypoint'][:T_other, :, :, :]
            else:
                annot_kpts[:annot['keypoint'].shape[0], :, :, :] = annot['keypoint']

            concat['keypoint'] =  np.concatenate((concat['keypoint'], annot_kpts), axis=2) # [T, 1, M, C]

        return concat

    
    def merge_samples(self, samples):
        # print(len(samples) , " samples to merge")
        origin = samples[0]
        original_view_dim = origin['keypoint'].shape[2]
        if self.sorted_by_pair: # TO BE VERIFIED
            for i in range(0, len(samples) - 1):
                first = samples[i]
                second = samples[i+1]

                assert 'keypoint' in first, str(first.keys())
                assert 'keypoint' in second, str(second.keys())

                if first['keypoint'].shape[0] > second['keypoint'].shape[0]:
                    origin = first
                    other = second
                else:
                    origin = second
                    other = first

                origin = self.merge_annot_into_other(other, origin)
        else:
            for i in range(1, len(samples)):
                origin = self.merge_annot_into_other(samples[i], origin, fit_annot_length=True)

        assert original_view_dim * len(samples) == origin['keypoint'].shape[2], f"Error in merging samples : {original_view_dim} * {len(samples)} != {origin['keypoint'].shape[2]}"

        return origin