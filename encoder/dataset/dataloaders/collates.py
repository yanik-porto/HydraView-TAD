from torch.utils.data.dataloader import default_collate
import numpy as np
import torch

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))

def TSU_collate_fn(batch):
    "Pads data and puts it into a tensor of same dimensions"

    # consider features, labels and others as input
    # output is list of [features, masks, labels, others]

    max_len = 0
    for b in batch:
        kpts = b['keypoint']
        if kpts.shape[0] > max_len:
            max_len = kpts.shape[0]

    new_batch = []
    for b in batch:
        f = np.zeros((max_len, b['keypoint'].shape[1], b['keypoint'].shape[2], b['keypoint'].shape[3]), np.float32)
        m = np.zeros((max_len), np.float32)
        l = np.zeros((max_len, b['window_labels'].shape[1]), np.float32)
        f[:b['keypoint'].shape[0]] = b['keypoint']
        m[:b['keypoint'].shape[0]] = 1
        l[:b['keypoint'].shape[0], :] = b['window_labels']
        input = {'keypoint': video_to_tensor(f), 'mask': torch.from_numpy(m), 'window_labels': torch.from_numpy(l)}
        if "duration" in b:
            input.update({"duration": b['duration']})

        new_batch.append(input)

    return default_collate(new_batch)

def MSTemba_TAD_collate_fn(batch, max_len=2500, label_key="tad_labels"):
    "Pads data and puts it into a tensor of same dimensions"

    # consider features, labels and others as input
    # output is list of [features, masks, labels, others]

    new_batch = []
    for b in batch:
        f = np.zeros((max_len, b['keypoint'].shape[1], b['keypoint'].shape[2], b['keypoint'].shape[3]), np.float32)
        m = np.zeros((max_len), np.float32)
        l = np.zeros((max_len, b[label_key].shape[1]), np.float32)
        f[:b['keypoint'].shape[0]] = b['keypoint']
        m[:b['keypoint'].shape[0]] = 1
        l[:b['keypoint'].shape[0], :] = b[label_key]
        input = {'keypoint': video_to_tensor(f), 'mask': torch.from_numpy(m), label_key: torch.from_numpy(l)}
        if "duration" in b:
            input.update({"duration": b['duration']})
        if "frame_dir" in b:
            input.update({"frame_dir": b['frame_dir']})

        new_batch.append(input)

    return default_collate(new_batch)

def Window_TAD_collate_fn(batch, max_len=250):
    # max_len = 250

    new_batch = []
    for b in batch:
        f = np.zeros((max_len,) + b['keypoint'].shape[1:], np.float32)
        m = np.zeros((max_len), np.float32)
        l = np.zeros((max_len, b['tad_labels'].shape[1]), np.float32)
        f[:b['keypoint'].shape[0]] = b['keypoint']
        m[:b['keypoint'].shape[0]] = 1
        l[:b['keypoint'].shape[0], :] = b['tad_labels']
        input = {'keypoint': f, 'mask': torch.from_numpy(m), 'tad_labels': torch.from_numpy(l)}
        if "duration" in b:
            input.update({"duration": b['duration']})
        if "frame_dir" in b:
            input.update({"frame_dir": b['frame_dir']})

        new_batch.append(input)

    return default_collate(new_batch)