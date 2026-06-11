import numpy as np

CONVENTION_NTU = {0:"base_of_the_spine", 1:"middle_of_the_spine", 2:"neck", 3:"head", 4:"left_shoulder", 5:"left_elbow", 6:"left_wrist", 7:"left_hand", 8:"right_shoulder", 
                    9:"right_elbow", 10:"right_wrist", 11:"right_hand", 12:"left_hip", 13:"left_knee", 14:"left_ankle", 15:"left_foot", 16:"right_hip", 17:"right_knee", 18:"right_ankle", 
                    19:"right_foot", 20:"Spine", 21:"Tip_of_the_left_hand", 22:"left_thumb", 23:"Tip_of_the_right_hand", 24:"right_thumb"}

CONVENTION_COCO = {0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear", 5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow", 9: "left_wrist", 10: "right_wrist", 11: "left_hip", 
                   12: "right_hip", 13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"}

def ntu_to_coco(ntu_keypoints):
    """
    Transform a vector of keypoints from the NTU convention to the COCO convention.

    Args:
        ntu_keypoints (numpy.ndarray): A NumPy array of keypoints in the NTU convention with shape [M, T, J, C].

    Returns:
        numpy.ndarray: A NumPy array of keypoints in the COCO convention with shape [M, T, J', C], where J' is the number of keypoints in the COCO convention.
    """
    M, T, J, C = ntu_keypoints.shape
    coco_keypoints = np.zeros((M, T, len(CONVENTION_COCO), C))

    for coco_idx, coco_part in CONVENTION_COCO.items():
        if coco_part in ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]:
            # Map to the NTU "head"
            ntu_part = "head"
        else:
            ntu_part = coco_part

        ntu_idx = None
        for idx, part in CONVENTION_NTU.items():
            if part == ntu_part:
                ntu_idx = idx
                break

        if ntu_idx is not None and ntu_idx < J:
            coco_keypoints[:, :, coco_idx, :] = ntu_keypoints[:, :, ntu_idx, :]
        else:
            print(f"Warning: {coco_part} not found in the NTU convention.")
            coco_keypoints[:, :, coco_idx, :] = 0

    return coco_keypoints

