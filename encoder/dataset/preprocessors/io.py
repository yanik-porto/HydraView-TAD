class LoadKineticsPose:
    """Load Kinetics Pose given filename (The format should be pickle)

    Required keys are "filename", "total_frames", "img_shape", "frame_inds",
    "anno_inds" (for mmpose source, optional), added or modified keys are
    "keypoint", "keypoint_score".

    Args:
        io_backend (str): IO backend where frames are stored. Default: 'disk'.
        squeeze (bool): Whether to remove frames with no human pose.
            Default: True.
        max_person (int): The max number of persons in a frame. Default: 10.
        keypoint_weight (dict): The weight of keypoints. We set the confidence
            score of a person as the weighted sum of confidence scores of each
            joint. Persons with low confidence scores are dropped (if exceed
            max_person). Default: dict(face=1, torso=2, limb=3).
        source (str): The sources of the keypoints used. Choices are 'mmpose'
            and 'openpose-18'. Default: 'mmpose'.
        kwargs (dict, optional): Arguments for FileClient.
    """

    def __init__(self,
                 io_backend='disk',
                 squeeze=True,
                 max_person=100,
                 keypoint_weight=dict(face=1, torso=2, limb=3),
                 source='mmpose',
                 **kwargs):

        self.io_backend = io_backend
        self.squeeze = squeeze
        self.max_person = max_person
        self.keypoint_weight = cp.deepcopy(keypoint_weight)
        self.source = source

        if source == 'openpose-18':
            self.kpsubset = dict(
                face=[0, 14, 15, 16, 17],
                torso=[1, 2, 8, 5, 11],
                limb=[3, 4, 6, 7, 9, 10, 12, 13])
        elif source == 'mmpose':
            self.kpsubset = dict(
                face=[0, 1, 2, 3, 4],
                torso=[5, 6, 11, 12],
                limb=[7, 8, 9, 10, 13, 14, 15, 16])
        else:
            raise NotImplementedError('Unknown source of Kinetics Pose')

        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):

        assert 'filename' in results
        filename = results.pop('filename')

        # only applicable to source == 'mmpose'
        anno_inds = None
        if 'anno_inds' in results:
            assert self.source == 'mmpose'
            anno_inds = results.pop('anno_inds')
        results.pop('box_score', None)

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        bytes = self.file_client.get(filename)

        # only the kp array is in the pickle file, each kp include x, y, score.
        kps = pickle.loads(bytes)

        total_frames = results['total_frames']

        frame_inds = results.pop('frame_inds')

        if anno_inds is not None:
            kps = kps[anno_inds]
            frame_inds = frame_inds[anno_inds]

        frame_inds = list(frame_inds)

        def mapinds(inds):
            uni = np.unique(inds)
            map_ = {x: i for i, x in enumerate(uni)}
            inds = [map_[x] for x in inds]
            return np.array(inds, dtype=np.int16)

        if self.squeeze:
            frame_inds = mapinds(frame_inds)
            total_frames = np.max(frame_inds) + 1

        # write it back
        results['total_frames'] = total_frames

        h, w = results['img_shape']
        if self.source == 'openpose-18':
            kps[:, :, 0] *= w
            kps[:, :, 1] *= h

        num_kp = kps.shape[1]
        num_person = mode(frame_inds)[-1][0]

        new_kp = np.zeros([num_person, total_frames, num_kp, 2],
                          dtype=np.float16)
        new_kpscore = np.zeros([num_person, total_frames, num_kp],
                               dtype=np.float16)
        # 32768 is enough
        num_person_frame = np.zeros([total_frames], dtype=np.int16)

        for frame_ind, kp in zip(frame_inds, kps):
            person_ind = num_person_frame[frame_ind]
            new_kp[person_ind, frame_ind] = kp[:, :2]
            new_kpscore[person_ind, frame_ind] = kp[:, 2]
            num_person_frame[frame_ind] += 1

        kpgrp = self.kpsubset
        weight = self.keypoint_weight
        results['num_person'] = num_person

        if num_person > self.max_person:
            for i in range(total_frames):
                np_frame = num_person_frame[i]
                val = new_kpscore[:np_frame, i]

                val = (
                    np.sum(val[:, kpgrp['face']], 1) * weight['face'] +
                    np.sum(val[:, kpgrp['torso']], 1) * weight['torso'] +
                    np.sum(val[:, kpgrp['limb']], 1) * weight['limb'])
                inds = sorted(range(np_frame), key=lambda x: -val[x])
                new_kpscore[:np_frame, i] = new_kpscore[inds, i]
                new_kp[:np_frame, i] = new_kp[inds, i]
            results['num_person'] = self.max_person

        results['keypoint'] = new_kp[:self.max_person]
        results['keypoint_score'] = new_kpscore[:self.max_person]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'squeeze={self.squeeze}, '
                    f'max_person={self.max_person}, '
                    f'keypoint_weight={self.keypoint_weight}, '
                    f'source={self.source}, '
                    f'kwargs={self.kwargs})')
        return repr_str