class FuseWindowDimension:
    """
    Fuse the window dimension into the person dimension.
    For input data of shape (num_windows, num_person, window_size, num_joints, 2 or 3),
    output data will be of shape (num_windows * num_person, window_size, num_joints, 2 or 3).
    """
    def __init__(self):
        pass

    def __call__(self, data_item):
        kpts = data_item['keypoint' if 'keypoint' in data_item else 'joints']  # shape (num_windows, num_person, window_size, num_joints, 2 or 3)
        num_windows = kpts.shape[0]
        num_person = kpts.shape[1]
        window_size = kpts.shape[2]
        num_joints = kpts.shape[3]
        coord_dim = kpts.shape[4]

        kpts_fused = kpts.reshape((num_windows * num_person, window_size, num_joints, coord_dim))
        data_item['keypoint' if 'keypoint' in data_item else 'joints'] = kpts_fused
        data_item['total_frames'] = window_size
        return data_item
    
class RetrieveWindowDimension:
    """
    Retrieve the window dimension from the person dimension.
    For input data of shape (num_windows * num_person, window_size, num_joints, 2 or 3),
    output data will be of shape (num_windows, num_person, window_size, num_joints, 2 or 3).
    """
    def __init__(self, num_person=1):
        self.num_person = num_person

    def __call__(self, data_item):
        kpts = data_item['keypoint' if 'keypoint' in data_item else 'joints']  # shape (num_windows * num_person, window_size, num_joints, 2 or 3)
        total_persons = kpts.shape[0]
        assert total_persons % self.num_person == 0, f"Total persons {total_persons} is not divisible by num_person {self.num_person}"
        num_windows = total_persons // self.num_person
        window_size = kpts.shape[1]
        num_joints = kpts.shape[2]
        coord_dim = kpts.shape[3]

        kpts_retrieved = kpts.reshape((num_windows, self.num_person, window_size, num_joints, coord_dim))
        data_item['keypoint' if 'keypoint' in data_item else 'joints'] = kpts_retrieved
        data_item['total_frames'] = window_size
        return data_item