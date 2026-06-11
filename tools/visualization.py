import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from collections import Counter
import pickle
import argparse
import os

def plot_temporal_actions(gt, pred, class_names=None, seq_number = 0):
    """
    gt   : array-like of shape [T, 1] or [T]
    pred : array-like of shape [T, C] (class probabilities)
    """

    gt = np.asarray(gt).squeeze()          # [T]
    pred = np.asarray(pred)                # [T, C]
    pred_labels = pred.argmax(axis=1)      # [T]
    T = gt.shape[0]
    C = pred.shape[1]


    if class_names is None:
        class_names = [f"class_{i}" for i in range(C)]

    if len(class_names) < C:
        class_names.append('none')

    # Define colors (edit if needed)
    colors = list(plt.get_cmap("Set3").colors[:C])
    if len(class_names) == C and len(colors) == C:
        colors[C-1] = (0.9, 0.9, 0.9)
    cmap = ListedColormap(colors)

    # Categorical normalization
    boundaries = np.arange(C + 1) - 0.5
    norm = BoundaryNorm(boundaries, C)

    # Prepare data for imshow (each row is one timeline)
    data = np.vstack([gt, pred_labels])

    fig, ax = plt.subplots(figsize=(12, 2))

    im = ax.imshow(
        data,
        aspect="auto",
        cmap=cmap,
        norm=norm, 
        interpolation="none"
    )

    # Y-axis labels
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["GT", "Ours"])

    # X-axis
    ax.set_xlabel("Time")

    # Legend
    handles = [
        plt.Line2D([0], [0], color=colors[i], lw=6)
        for i in range(C)
    ]
    ax.legend(
        handles,
        class_names,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.4),
        ncol=C
    )

    plt.title(f'Sequence #{seq_number}')
    plt.tight_layout()
    plt.show()


def plot_temporal_actions_capped(gt, pred, class_names=None, seq_number = 0, M=3):

    gt = np.asarray(gt).squeeze()          # [T]
    other_gt_name="others"
    other_pred_name="others"
    other_gt_color=(0.6, 0.6, 0.6)
    other_pred_color=(0.6, 0.6, 0.6)

    pred = np.asarray(pred)                # [T, C]
    pred_labels = pred.argmax(axis=1)      # [T]

    T = gt.shape[0]

    C = pred.shape[1]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(C)]
    if len(class_names) < C:
        class_names.append('none')

    # ---- 1. Select top-M GT classes by frequency ----
    kept_gt_classes = range(C)
    lastIsNone = True
    if C > M:
        lastIsNone = False
        gt_counts = Counter(gt.tolist())
        kept_gt_classes = [c for c, _ in gt_counts.most_common(M)]
        # breakpoint()
        if C-1 in kept_gt_classes:
            kept_gt_classes.sort()
            # kept_gt_classes.append(kept_gt_classes.pop(C-1)) # place none at the end of the list
        # if kept_gt_classes[0] == C-1:
        #     kept_gt_classes = kept_gt_classes[1:] + [kept_gt_classes[0]]
            lastIsNone = True
 
    # ---- 2. Build remapping ----
    gt_map = {c: i for i, c in enumerate(kept_gt_classes)}
    GT_OTHER_ID = len(gt_map)

    mapped_gt = np.array([
        gt_map[c] if c in gt_map else GT_OTHER_ID
        for c in gt
    ])

    # ---- 3. Remap predictions ----
    mapped_pred = np.array([
        gt_map[p] if p in gt_map else GT_OTHER_ID + 1
        for p in pred_labels
    ])

    PRED_OTHER_ID = GT_OTHER_ID + 1
    total_classes = PRED_OTHER_ID + 1

    # ---- 4. Colors ----
    base_colors = plt.get_cmap("Set3").colors
    base_colors = list(base_colors[:len(gt_map)])
    if lastIsNone:
        base_colors[-1] = (0.9, 0.9, 0.9)
    
    colors = (
        base_colors +
        [other_gt_color] +
        [other_pred_color]
    )

    cmap = ListedColormap(colors)

    # ---- 5. Categorical normalization ----
    boundaries = np.arange(total_classes + 1) - 0.5
    norm = BoundaryNorm(boundaries, total_classes)

    data = np.vstack([mapped_gt, mapped_pred])

    # ---- 6. Plot ----
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.imshow(
        data,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest"
    )

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["GT", "Prediction"])
    ax.set_xlabel("Windows")

    # ---- 7. Legend ----
    legend_names = (
        [class_names[c] if class_names else f"class_{c}" for c in kept_gt_classes]
        # [other_gt_name, other_pred_name]
    )

    if C > M:
        legend_names += [other_gt_name]

    handles = [
        plt.Line2D([0], [0], color=colors[i], lw=6)
        for i in range(total_classes)
    ]

    ax.legend(
        handles,
        legend_names,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.5),
        ncol=min(4, total_classes)
    )
    plt.title(f'Sequence #{seq_number}')

    plt.tight_layout()
    plt.show()
