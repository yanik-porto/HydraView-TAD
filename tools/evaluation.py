import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if type(output) is tuple:
        output = output[0]

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_multiple_labels(output, targets, topk=(1,)):
    assert output.shape[1] >= targets.shape[1], f"Predictions shape {output.shape} must match targets shape {targets.shape}"

    output = output[:, :targets.shape[1]]

    maxk = max(topk)
    _, preds = output.topk(maxk, 1, True, True)

    batch_size = len(output)
    with torch.no_grad():
        res = [[] for _ in range(len(topk))]

        for pred, target in zip(preds, targets):
            target_nz = torch.nonzero(target)
            for i, k in enumerate(topk):
                correct = any(p in target_nz for p in  pred[:k])
                res[i].append(correct)
    return [sum(r) / len(r) * 100. for r in res]

def map_gtidx_to_color(feats_gt_idxs, label_map):
    # get unique ids
    onlyidxs = sorted(set(feats_gt_idxs))

    # color idxs with labels
    coloridxs = list(range(len(onlyidxs)))
    colorlabels = [label_map[idx] for idx in onlyidxs]

    # labels as color index
    gt_to_color = dict(zip(onlyidxs, coloridxs))
    all_color_idxs = [gt_to_color[gt_idx] for gt_idx in feats_gt_idxs]
    return all_color_idxs, colorlabels

def top_k_by_action(scores, labels, k=1):
    labels = np.array(labels)[:, np.newaxis]
    max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]

    match_by_action = {}

    for i in range(len(labels)):
        label = labels[i][0]
        pred = max_k_preds[i]

        if label not in match_by_action.keys():
            match_by_action[label] = []
        match_by_action[label].append(np.logical_or.reduce(pred == np.array(label)))

    topk_by_action = {}
    for action in match_by_action.keys():
        topk_by_action[action] = sum(match_by_action[action]) / len(match_by_action[action])

    return topk_by_action

def cosine_loss(pred, target):
    """
    pred and target are both [batch_size, 2], normalized to unit vectors.
    """
    return 1 - F.cosine_similarity(pred, target, dim=1).mean()

def mse_vector_loss(pred, target):
    return F.mse_loss(pred, target)

def tad_accuracy(output, target):
    outputs_final, mask = output
    labels, duration = target

    if outputs_final.dim() > 3:
        outputs_final = outputs_final[0,:,:,:]
    if outputs_final.shape[1] != labels.shape[1]: # not needed in ms-temba for example
        outputs_final = outputs_final.permute(0, 2, 1)

    probs_f = F.sigmoid(outputs_final) * mask.unsqueeze(2)
    fps = outputs_final.size()[1] / duration[0]
    return probs_f


