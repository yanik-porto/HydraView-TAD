import argparse
from torch.utils.data import DataLoader
from dvclive import Live
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm 
import importlib
from functools import partial
from statistics import mean

from encoder.encoders import *
from encoder.dataset.datasets import *
from heads import *
from encoder.dataset.tools.config import load_config
from encoder.dataset.tools.measure import AverageMeter, APMeter
from tools.evaluation import accuracy, map_gtidx_to_color, accuracy_multiple_labels, top_k_by_action, tad_accuracy
from tools.evaluation_det import getSingleStreamDetectionMAP
from tools.visualization import plot_temporal_actions, plot_temporal_actions_capped
from tools.checkpoint import clean_checkpoint
from encoder.dataset.dataloaders.formater import split_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Test motion head")
    parser.add_argument("config", type=str, help="Path to configuration file")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument('--params', type=str, default=None, help='set if some parameters should be loaded from yaml')
    parser.add_argument('--save_metric', action="store_true", default=False, help="set if metric has to be saved in dvc")
    parser.add_argument('--name', type=str, default="testset", help="name of the dataset to load in config")
    parser.add_argument('--gpu_id_eval', type=int, default=0, help="Id of the gpu where running evaluation (and trining in case of single gpu)")
    parser.add_argument('--gt_name', type=str, default='label', help="Name of the gt variable to compare the output with")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config, args.params, update_default=False)

    # create model
    print("* create model ... ", end="")
    encoder = create_encoder(config)
    head = create_head(config)
    device = torch.device(args.gpu_id_eval)
    encoder, head = encoder.to(device), head.to(device)
    print("done")

    # load checkpoint
    print("* load checkpoint ...", end="")
    checkpoint = torch.load(args.checkpoint, map_location='cuda:' + str(args.gpu_id_eval))#, weights_only=False)
    if "epoch" in checkpoint:
        print("from epoch #", checkpoint["epoch"], " ... ", end="")
    chkpt_encoder = clean_checkpoint(checkpoint['encoder'], remove_fc=True)
    chkpt_head = clean_checkpoint(checkpoint['head'])
    encoder.load_state_dict(chkpt_encoder, strict=True)
    head.load_state_dict(chkpt_head, strict=True)
    print("done")

    # load testing dataset
    print("* create dataset ...", end="")
    dataset = create_dataset(config, setname=args.name)
    dl_settings = config["test"]["dataloader"]
    if 'collate_fn' in dl_settings:
        collate_class_name = dl_settings['collate_fn']
        collate_module_name = 'encoder.dataset.dataloaders.collates'
        collate_module = importlib.import_module(collate_module_name)
        collate_fn = getattr(collate_module, collate_class_name)
        if 'collate_params' in dl_settings:
            collate_fn = partial(collate_fn, **dl_settings['collate_params'])
            del dl_settings["collate_params"]
        dl_settings["collate_fn"] = collate_fn
    dl = DataLoader(dataset, **dl_settings)
    print(" done : ", len(dataset), " inputs")

    # test
    print("***************")
    print("**** test ****")
    print("Run ", len(dl), " iters")
    encoder.eval()
    head.eval()

    # create metrics
    metrics = {}
    if args.gt_name == "window_labels":
        metrics["Top1"] = APMeter()
    elif args.gt_name == "tad_labels":
        metrics["mAP@0.1"] = AverageMeter()
        metrics["mAP@0.25"] = AverageMeter()
        metrics["mAP@0.5"] = AverageMeter()
    else:
        metrics["Top1"] = AverageMeter()
        metrics["Top5"] = AverageMeter()
    top1_ml = AverageMeter()
    top5_ml = AverageMeter()
    all_outputs = []
    all_gts = []
    all_feats = []

    avg_meter_by_class = {}


    # iter on dataset
    for ii, batch in tqdm(enumerate(dl)):
        keypoints, labels = split_batch(batch, device, args.gt_name)

        feats = encoder(keypoints)

        motion_logits = head(feats)

        if args.gt_name == "tad_labels":
            bin_labels = labels[0].cpu() if type(labels) is tuple else labels.cpu()
            gt = torch.argmax(bin_labels, axis=2).numpy()
            mask = motion_logits[1].to(bool)
            preds = motion_logits[0][mask]
            gt = gt[mask.cpu().numpy()]
            preds_viz = torch.sigmoid(preds).data.cpu().numpy()
            seq_name = ii if len(labels) < 3 else labels[2]
            # plot_temporal_actions_capped(gt, preds_viz, dl.dataset.label_map, seq_name, M=6)
            dmap_list, iou_list, aps_by_class = getSingleStreamDetectionMAP(preds, gt, multi=True)
            for ckey in aps_by_class[0]: # all aps for iou 50%
                if ckey not in avg_meter_by_class:
                    avg_meter_by_class[ckey] = AverageMeter()
                mean_class = mean(aps_by_class[0][ckey])
                avg_meter_by_class[ckey].update(mean_class)
            metrics["mAP@0.1"].update(dmap_list[0], 1) # only labels needed, no duration
            metrics["mAP@0.25"].update(dmap_list[1], 1) # only labels needed, no duration
            metrics["mAP@0.5"].update(dmap_list[2], 1) # only labels needed, no duration
        elif args.gt_name == "window_labels":
            probs = tad_accuracy(motion_logits, labels)
            metrics["Top1"].update(probs.data.cpu().numpy()[0], bin_labels.numpy()[0]) # only labels needed, no duration
        else:
            if args.gt_name == "label":
                if type(motion_logits) is tuple:
                    motion_logits = motion_logits[0]
                if len(motion_logits.shape) == 3:
                    M = motion_logits.shape[1]
                    motion_logits = motion_logits.flatten(0, 1)
                    labels = labels.repeat_interleave(M)
                acc1, acc5 = accuracy(motion_logits, labels, topk=(1, 5))

            metrics["Top1"].update(acc1[0].detach().cpu().numpy(), len(batch['keypoint']))
            metrics["Top5"].update(acc5[0].detach().cpu().numpy(), len(batch['keypoint']))
        output = motion_logits

        if 'binary_labels' in batch:
            acc1_ml, acc5_ml = accuracy_multiple_labels(output.detach().cpu(), batch['binary_labels'], topk=(1, 5))
            top1_ml.update(acc1_ml, len(batch['keypoint']))
            top5_ml.update(acc5_ml, len(batch['keypoint']))

        if type(output) is tuple:
            output = output[0]
        if type(labels) is tuple:
            labels = labels[0]

        if args.gt_name not in ("tad_labels", "window_labels"):
            all_outputs.extend(output.detach().cpu().numpy())
            all_gts.extend(labels.detach().cpu().numpy())

        if args.save_metric:
            feats = feats.detach().cpu().numpy()
            all_feats.append(feats[0, 0:1].reshape(-1))

            if len(all_feats) >= 5000:
                break

        output = None


    for metric_name, metric_value in metrics.items():
        print(f" | {metric_name}: {metric_value.value():.2f}", end="")
    print("")
    if top1_ml.count > 0:
        print("Accuracy Top1 Multi Labels : ", top1_ml.value())
        print("Accuracy Top5 Multi Labels : ", top5_ml.value())

    if args.gt_name not in ("tad_labels", "window_labels"):
        np.set_printoptions(legacy='1.25')
        top1_by_action  = top_k_by_action(np.array(all_outputs), all_gts)
        print(dict(sorted(top1_by_action.items())).values())
        # print(dict(sorted(top1_by_action.items())))

    elif args.gt_name in ("tad_labels"):
        print([(int(key), val.value()) for key, val in avg_meter_by_class.items()])

    if args.save_metric:
        with Live("artifacts/metrics_" + args.name) as live:
            for metric_name, metric_value in metrics.items():
                live.log_metric(metric_name, metric_value.avg)

            # build confusion matrix
            pred = list(np.argmax(np.array(all_outputs), axis=1))
            all_gt_idxs = all_gts
            if max(pred) < len(dataset.label_map) and max(all_gts) < len(dataset.label_map):
                pred = [dataset.label_map[p] for p in pred]
                all_gts = [dataset.label_map[l] for l in all_gts]
            # live.log_sklearn_plot(
            #     "confusion_matrix", all_gts, pred, name="cm.json")

            # build tsne
            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(np.array(all_feats))
            fig, ax = plt.subplots(1,1)
            all_color_idxs, colorlabels = map_gtidx_to_color(all_gt_idxs[:len(all_feats)], dataset.label_map)
            scatter_motion = plt.scatter(X_tsne[:,0], X_tsne[:,1], c=all_color_idxs, marker='.', cmap='gist_rainbow', linewidths=0.5)
            handles, _ = scatter_motion.legend_elements(prop='colors')
            plt.legend(handles, colorlabels)

            plt.legend()
            live.log_image("tsne.png", fig)
            print("metrics saved in ", "artifacts/metrics_" + args.name)
