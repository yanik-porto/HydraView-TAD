#This is the evaluation code from SW-TAL paper in the repo : https://github.com/line/Skeleton-Temporal-Action-Localization

import pickle
from collections import Counter

import numpy as np
import torch

def str2ind(categoryname, classlist):
    return [i for i in range(len(classlist)) if categoryname == classlist[i]][0]

def get_segments(scores, activity_threshold):
  """Get prediction segments of a video."""
  # Each segment contains start, end, class, confidence score.
  # Sum of all probabilities (1 - probability of no-activity)
  # activity_prob = 1 - scores[:, 0]
  activity_prob = 1 - scores[:, -1]
  # Binary vector indicating whether a clip is an activity or no-activity
  activity_tag = np.zeros(activity_prob.shape, dtype=np.int32)
  activity_tag[activity_prob >= activity_threshold] = 1
  assert activity_tag.ndim == 1
  # For each index, subtract the previous index, getting -1, 0, or 1
  # 1 indicates the start of a segment, and -1 indicates the end.
  padded = np.pad(activity_tag, pad_width=1, mode='constant')
  diff = padded[1:] - padded[:-1]
  indexes = np.arange(diff.size)
  startings = indexes[diff == 1]
  endings = indexes[diff == -1]
  assert startings.size == endings.size

  segments = []
  for start, end in zip(startings, endings):
    segment_scores = scores[start:end, :]
    class_prob = np.mean(segment_scores, axis=0)
    # segment_class_index = np.argmax(class_prob[1:]) + 1
    segment_class_index = np.argmax(class_prob[:-1])
    confidence = np.mean(segment_scores[:, segment_class_index])
    # if confidence < 0.5:
    #   continue
    seg = [segment_class_index, start, end, confidence]
    segments.append(np.array(seg))

  segments_by_class = []
  n_classes = scores.shape[1]
  for class_index in range(n_classes):
        class_segments = [
            seg for seg in segments if seg[0] == class_index
        ]
        segments_by_class.append(np.array(class_segments))

  return segments_by_class


def encode_mask_to_rle(mask):
    """
    mask: numpy array binary mask
    1 - mask
    0 - background
    Returns encoded run length
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return runs


def getActLoc(
    frm_preds, act_thresh_cas, annot_by_frame, num_classes=4, multi=False
):


    # Build ground truth segments
    if multi:
        gtsegments = []
        gtlabels = []
        for idx in range(len(annot_by_frame)):
            gt = annot_by_frame[idx]
            gt_ = set(gt)
            gt_.discard(num_classes)
            gts = []
            gtl = []
            for c in list(gt_):
                gt_encoded = encode_mask_to_rle(gt == c)
                gts.extend(
                    [
                        [x - 1, x + y - 2]
                        for x, y in zip(gt_encoded[::2], gt_encoded[1::2])
                    ]
                )
                gtl.extend([c for item in gt_encoded[::2]])
            gtsegments.append(gts)
            gtlabels.append(gtl)
    else:
        gtsegments = []
        # gtlabels = []
        for idx in range(len(annot_by_frame)):
            gt = annot_by_frame[idx]
            gt_encoded = encode_mask_to_rle(gt)
            gtsegments.append(
                [[x - 1, x + y - 2] for x, y in zip(gt_encoded[::2], gt_encoded[1::2])]
            )
            # gtlabels.append([data["Y"][idx] for item in gt_encoded[::2]])

    # videoname = np.array(data["sid"])

    # keep ground truth and predictions for instances with temporal annotations
    # gtl, vn, vp, fp, vl = [], [], [], [], []
    gtl, vn, fp, vl = [], [], [], []
    for i, s in enumerate(gtsegments):
        if len(s):
            # gtl.append(gtlabels[i])
            # vn.append(videoname[i])
            # vp.append(vid_preds[i])
            fp.append(frm_preds[i])
            # vl.append(vid_lens[i])
    # gtlabels = gtl
    # videoname = vn

    # which categories have temporal labels ?
    # templabelidx = sorted(list(set([l for gtl in gtlabels for l in gtl])))

    dataset_segment_predict = []
    # class_threshold = args.class_threshold
    for c in range(frm_preds[0].shape[1]):
        c_temp = []
        # Get list of all predictions for class c
        for i in range(len(fp)):
            # vid_cls_score = vp[i][c]
            vid_cas = fp[i][:, c]
            vid_cls_proposal = []
            # if vid_cls_score < class_threshold:
            #     continue
            for t in range(len(act_thresh_cas)):
                thres = act_thresh_cas[t]
                vid_pred = np.concatenate(
                    [np.zeros(1), (vid_cas > thres).astype("float32"), np.zeros(1)],
                    axis=0,
                )
                vid_pred_diff = [
                    vid_pred[idt] - vid_pred[idt - 1] for idt in range(1, len(vid_pred))
                ]
                s = [idk for idk, item in enumerate(vid_pred_diff) if item == 1]
                e = [idk for idk, item in enumerate(vid_pred_diff) if item == -1]
                for j in range(len(s)):
                    len_proposal = e[j] - s[j]
                    if len_proposal >= 3:
                        inner_score = np.mean(vid_cas[s[j] : e[j] + 1])
                        outer_s = max(0, int(s[j] - 0.25 * len_proposal))
                        outer_e = min(
                            int(vid_cas.shape[0] - 1),
                            int(e[j] + 0.25 * len_proposal + 1),
                        )
                        outer_temp_list = list(range(outer_s, int(s[j]))) + list(
                            range(int(e[j] + 1), outer_e)
                        )
                        if len(outer_temp_list) == 0:
                            outer_score = 0
                        else:
                            outer_score = np.mean(vid_cas[outer_temp_list])
                        c_score = inner_score - 0.6 * outer_score
                        vid_cls_proposal.append([i, s[j], e[j] + 1, c_score])
            pick_idx = NonMaximumSuppression(np.array(vid_cls_proposal), 0.2)
            nms_vid_cls_proposal = [vid_cls_proposal[k] for k in pick_idx]

            c_temp += nms_vid_cls_proposal
        if len(c_temp) > 0:
            c_temp = np.array(c_temp)
        dataset_segment_predict.append(c_temp)
    """
    for i, pred in enumerate(dataset_segment_predict):
        print (f"#{i} class {c} has {len(pred)} predictions")
    """
    return dataset_segment_predict


def NonMaximumSuppression(segs, overlapThresh):
    # if there are no boxes, return an empty list
    if len(segs) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if segs.dtype.kind == "i":
        segs = segs.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the segments
    s = segs[:, 1]
    e = segs[:, 2]
    scores = segs[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the score of the bounding box
    area = e - s + 1
    idxs = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest coordinates for the start of
        # the segments and the smallest coordinates
        # for the end of the segments
        maxs = np.maximum(s[i], s[idxs[:last]])
        mine = np.minimum(e[i], e[idxs[:last]])

        # compute the length of the overlapping area
        l = np.maximum(0, mine - maxs + 1)
        # compute the ratio of overlap
        overlap = l / area[idxs[:last]]

        # delete segments beyond the threshold
        idxs = np.delete(
            idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        )
    return pick


def getLocMAP(seg_preds, th, annot_by_frame, num_classes=4, multi=False, factor=1.0):
  
    if multi:
        gtsegments = []
        gtlabels = []
        # for idx in range(len(data["L"])):
        for idx in range(len(annot_by_frame)):
            gt = annot_by_frame[idx]
            gt_ = set(gt)
            gt_.discard(num_classes)
            gts = []
            gtl = []
            for c in list(gt_):
                gt_encoded = encode_mask_to_rle(gt == c)
                gts.extend(
                    [
                        [x - 1, x + y - 2]
                        for x, y in zip(gt_encoded[::2], gt_encoded[1::2])
                    ]
                )
                gtl.extend([c for item in gt_encoded[::2]])
            gtsegments.append(gts)
            gtlabels.append(gtl)
    else:
        gtsegments = []
        gtlabels = []
        for idx in range(len(annot_by_frame)):
            gt = annot_by_frame[idx]
            gt_encoded = encode_mask_to_rle(gt)
            gtsegments.append(
                [[x - 1, x + y - 2] for x, y in zip(gt_encoded[::2], gt_encoded[1::2])]
            )
            gtlabels.append([annot_by_frame[idx] for item in gt_encoded[::2]])
            # gtlabels.append([data["Y"][idx] for item in gt_encoded[::2]])

    # videoname = np.array(data["sid"])
    """
    cnt = Counter(data['Y'])
    d = cnt.most_common()
    print (d)
    """
    # which categories have temporal labels ?
    templabelidx = sorted(list(set([l for gtl in gtlabels for l in gtl])))

    ap = []
    ap_by_class = {}
    for c in templabelidx:
        segment_predict = seg_preds[c]
        # Sort the list of predictions for class c based on score
        if len(segment_predict) == 0:
            ap.append(0.0)
            continue
        segment_predict = segment_predict[np.argsort(-segment_predict[:, 3])]

        # Create gt list
        segment_gt = [
            [i, gtsegments[i][j][0], gtsegments[i][j][1]]
            for i in range(len(gtsegments))
            for j in range(len(gtsegments[i]))
            if gtlabels[i][j] == c
        ]
        gtpos = len(segment_gt)

        # Compare predictions and gt
        tp, fp = [], []
        for i in range(len(segment_predict)):
            matched = False
            best_iou = 0
            for j in range(len(segment_gt)):
                if segment_predict[i][0] == segment_gt[j][0]:
                    gt = range(
                        int(round(segment_gt[j][1] * factor)),
                        int(round(segment_gt[j][2] * factor)),
                    )
                    p = range(int(segment_predict[i][1]), int(segment_predict[i][2]))
                    IoU = float(len(set(gt).intersection(set(p)))) / float(
                        len(set(gt).union(set(p)))
                    )
                    if IoU >= th:
                        matched = True
                        if IoU > best_iou:
                            best_iou = IoU
                            best_j = j
            if matched:
                del segment_gt[best_j]
            tp.append(float(matched))
            fp.append(1.0 - float(matched))
        tp_c = np.cumsum(tp)
        fp_c = np.cumsum(fp)
        # print (c, tp, fp)
        if sum(tp) == 0:
            prc = 0.0
        else:
            cur_prec = tp_c / (fp_c + tp_c)
            cur_rec = tp_c / gtpos
            prc = _ap_from_pr(cur_prec, cur_rec)
        ap.append(prc)
        if not c in ap_by_class:
            ap_by_class[c] = []
        ap_by_class[int(c)].append(float(prc))

    # print(f" ".join([f"{item*100:.2f}" for item in ap]))
    if ap:
        return 100 * np.mean(ap), ap_by_class
    else:
        return 0


# Inspired by Pascal VOC evaluation tool.
def _ap_from_pr(prec, rec):
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])

    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])

    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])

    return ap


def compute_iou(dur1, dur2):
    # find the each edge of intersect rectangle
    left_line = max(dur1[0], dur2[0])
    right_line = min(dur1[1], dur2[1])

    # judge if there is an intersect
    if left_line >= right_line:
        return 0
    else:
        intersect = right_line - left_line
        union = max(dur1[1], dur2[1]) - min(dur1[0], dur2[0])
        return intersect / union


def getSingleStreamDetectionMAP(
    output, annot_by_frame, args=None, multi=False, factor=1.0
):
    #format input data
    if type(output) is tuple:
        mask = output[1].to(bool)
        preds = output[0]

        if preds.dim() > 3:
            preds = preds[0,:,:,:]
        if preds.shape[1] != annot_by_frame.shape[1]: # not needed in ms-temba for example
            preds = preds.permute(0, 2, 1)

        frm_preds = preds[mask]

        annot_by_frame = annot_by_frame[mask.cpu().numpy()]
    else:
        frm_preds = output

    # normalize the predictions
    frm_preds = torch.sigmoid(frm_preds)

    frame_preds = frm_preds.unsqueeze(dim=0).cpu().detach().numpy() #if len(frm_preds.shape) == 2 else frm_preds[0]
    ann_by_frame = np.expand_dims(annot_by_frame, axis=0)# if len(annot_by_frame.shape) == 2 else annot_by_frame[0]


    start_threshold = 0.03
    end_threshold = 0.055
    threshold_interval = 0.005
    num_classes = frm_preds.shape[1] - 1
    # # iou_list = [0.1, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]
    iou_list = [0.1, 0.25, 0.5]
    seg = getActLoc(
        frame_preds,
        np.arange(start_threshold, end_threshold, threshold_interval),
        ann_by_frame,
        num_classes,
        multi=multi,
    )

    # seg = get_segments(frame_preds[0], 0.2)

    dmap_list = []
    aps_by_class = []
    for iou in iou_list:
        ap_mean, ap_by_class = getLocMAP(seg, iou, ann_by_frame, num_classes, multi=multi, factor=factor)
        dmap_list.append(ap_mean)
        aps_by_class.append(ap_by_class)

    return dmap_list, iou_list, aps_by_class