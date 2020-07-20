import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import motmetrics as mm
from eval_utils import calculate_iou, decode_rle, compute_mot_metrics


parser = argparse.ArgumentParser()
parser.add_argument('--gt_file', help='Path to file with ground truth annotations.', type=str)
parser.add_argument('--pred_file', help='Path to file with prediction annotations.', type=str)
parser.add_argument('--results_path', help='Path to output file.', type=str)
parser.add_argument('--start_step', help='Step to start evaluation from (2 for next-step prediction).', type=int, default=2)
parser.add_argument('--stop_step', help='Step to start evaluation from (10 for next-step prediction).', type=int, default=10)
parser.add_argument('--iou_thresh', help='IoU threshold.', type=float, default=0.5)
parser.add_argument('--im_size', help='Image size.', type=int, default=64)
parser.add_argument('--min_num_pix', help='Minimum number of pixels for valid mask.', type=int, default=5)
parser.add_argument('--exclude_bg', help='Exclude background masks from evaluation.', action='store_true')


def exclude_bg(dists, gt_ids, pred_ids, n_gt_bg):
    # remove background slots
    gt_idx = -1
    for k in range(n_gt_bg):
        if dists.shape[1] > 0:
            pred_bg_id = np.where(dists[gt_idx] > 0.2)[0]
            dists = np.delete(dists, pred_bg_id, 1)
            pred_ids = [pi for l,pi in enumerate(pred_ids) if not l in pred_bg_id]
        dists = np.delete(dists, gt_idx, 0)  
        del gt_ids[gt_idx]   
    return dists, gt_ids, pred_ids



def compute_dists_per_frame(gt_frame, pred_frame, args):
    # Compute pairwise distances between gt objects and predictions per frame. 
    s = args.im_size
    n_pred = len(pred_frame['ids'])
    n_gt = len(gt_frame['ids'])

    # accumulate pred masks for frame
    preds = []
    pred_ids = []
    for j in range(n_pred):
        mask = decode_rle(pred_frame['masks'][j], (s, s))
        if mask.sum() > args.min_num_pix:
            preds.append(mask)
            pred_ids.append(pred_frame['ids'][j])
    preds = np.array(preds)

    # accumulate gt masks for frame
    gts = []
    gt_ids = []
    for h in range(n_gt):
        mask = decode_rle(gt_frame['masks'][h], (s, s))
        if mask.sum() > args.min_num_pix:
            gts.append(mask)
            gt_ids.append(gt_frame['ids'][h])
    gts = np.array(gts)

    # compute pairwise distances
    dists = np.ones((len(gts), len(preds)))
    for h in range(len(gts)):
        for j in range(len(preds)): 
            dists[h, j] = calculate_iou(gts[h], preds[j])

    if args.exclude_bg:
        n_gt_bg = gt_frame['num_bg']
        dists, gt_ids, pred_ids = exclude_bg(dists, gt_ids, pred_ids, n_gt_bg)
        
    dists = 1. - dists
    dists[dists > args.iou_thresh] = np.nan
        
    return dists, gt_ids, pred_ids
        


def accumulate_events(gt_dict, pred_dict, args):
    acc = mm.MOTAccumulator()
    count = 0
    for i in tqdm(range(len(gt_dict))):
        for t in range(args.start_step, args.stop_step):
            gt_dict_frame = gt_dict[i][t]
            pred_dict_frame = pred_dict[i][t]
            dist, gt_ids, pred_ids = compute_dists_per_frame(gt_dict_frame, pred_dict_frame, args)
            acc.update(gt_ids, pred_ids, dist, frameid=count)
            count += 1
    return acc



def main(args):
    # load json files
    with open(args.pred_file, 'r') as file:
        pred_dict = json.load(file)
    with open(args.gt_file, 'r') as file:
        gt_dict = json.load(file) 
    assert len(pred_dict) == len(gt_dict)

    if args.exclude_bg:
        print('Excluding background slots')

    print('Acuumulate events.')
    acc = accumulate_events(gt_dict, pred_dict, args)

    print('Accumulate metrics.')
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp', 'num_matches', 'num_switches', 'num_false_positives', 'num_misses', 'num_objects'], name='acc')

    # compute tracking metrics
    metrics = compute_mot_metrics(acc, summary)
                  
    print(metrics)
    metrics = {key: value['acc'] for key, value in metrics.items()}
    metrics = pd.Series(metrics).to_json()

    print('Saving results to {}'.format(args.results_path))
    with open(args.results_path, 'w') as f:
        json.dump(metrics, f)
        
    print('Done.')



if __name__ == '__main__':

    args = parser.parse_args()
    main(args)
