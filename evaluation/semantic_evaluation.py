# Copyright 2021 Toyota Research Institute.  All rights reserved.

import argparse
import os
from argparse import Namespace
from collections import OrderedDict
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

ddad_to_cityscapes = {
    # ROAD
    7:    7,    # Crosswalk
    10:   7,    # LaneMarking
    11:   7,    # LimitLine
    13:   7,    # OtherDriveableSurface
    21:   7,    # Road
    24:   7,    # RoadMarking
    27:   7,    # TemporaryConstructionObject
    # SIDEWALK
    25:   8,    # SideWalk
    23:   8,    # RoadBoundary (Curb)
    14:   8,    # OtherFixedStructure
    15:   8,    # OtherMovable
    # WALL
    16:  12,    # Overpass/Bridge/Tunnel
    22:  12,    # RoadBarriers
    # FENCE
    8:   13,    # Fence
    # BUILDING
    2:   11,    # Building
    # POLE
    9:   17,    # HorizontalPole
    35:  17,    # VerticalPole
    # TRAFFIC LIGHT
    30:  19,    # TrafficLight
    # TRAFFIC SIGN
    31:  20,    # TrafficSign
    # VEGETATION
    34:  21,    # Vegetation
    # TERRAIN
    28:  22,    # Terrain
    # SKY
    26:  23,    # Sky
    # PERSON
    18:  24,    # Pedestrian
    # RIDER
    20:  25,    # Rider
    # CAR
    4:   26,    # Car
    # TRUCK
    33:  27,    # Truck
    5:   27,    # Caravan/RV
    6:   27,    # ConstructionVehicle
    # BUS
    3:   28,    # Bus
    # TRAIN
    32:  31,    # Train
    # MOTORCYCLE
    12:  32,    # Motorcycle
    # BICYCLE
    1:   33,    # Bicycle
    # IGNORE
    0:  255,    # Animal
    17: 255,    # OwnCar (EgoCar)
    19: 255,    # Railway
    29: 255,    # TowedObject
    36: 255,    # WheeledSlow
    37: 255,    # Void
}

map_classes = {
    "Road": 7,
    "Sidewalk": 8,
    "Wall": 12,
    "Fence": 13,
    "Building": 11,
    "Pole": 17,
    "T.Light": 19,
    "T.Sign": 20,
    "Vegetation": 21,
    "Terrain": 22,
    "Sky": 23,
    "Person": 24,
    "Rider": 25,
    "Car": 26,
    "Truck": 27,
    "Bus": 28,
    "Train": 31,
    "Motorcycle": 32,
    "Bicycle": 33,
    "Ignore": 255,
}


def convert_ontology(semantic_id, ontology_convert):
    """Convert from one ontology to another"""
    if ontology_convert is None:
        return semantic_id
    else:
        semantic_id_convert = semantic_id.clone()
        for key, val in ontology_convert.items():
            semantic_id_convert[semantic_id == key] = val
        return semantic_id_convert


def parse_args():
    """Parse arguments for benchmark script"""
    parser = argparse.ArgumentParser(description='PackNet-SfM benchmark script')
    parser.add_argument('--gt_folder', type=str,
                        help='Folder containing predicted depth maps (.npz with key "depth")')
    parser.add_argument('--pred_folder', type=str,
                        help='Folder containing predicted depth maps (.npz with key "depth")')
    parser.add_argument('--output_folder', type=str,
                        help='Output folder where information will be stored')
    parser.add_argument('--use_gt_scale', action='store_true',
                        help='Use ground-truth median scaling on predicted depth maps')
    parser.add_argument('--ranges', type=float, nargs='+', default=[200],
                        help='Depth ranges to consider during evaluation')
    parser.add_argument('--classes', type=str, nargs='+', default=['All', 'Car', 'Pedestrian'],
                        help='Semantic classes to consider during evaluation')
    parser.add_argument('--metric', type=str, default='rmse', choices=['abs_rel', 'rmse', 'silog', 'a1'],
                        help='Which metric will be used for evaluation')
    parser.add_argument('--min_num_valid_pixels', type=int, default=1,
                        help='Minimum number of valid pixels to consider')
    args = parser.parse_args()
    return args


def create_summary_table(ranges, classes, matrix, folder, metric):

    # Prepare variables
    title = "Semantic/Range Depth Evaluation (%s) -- {}" % metric.upper()
    ranges = ['{}m'.format(r) for r in ranges]
    result = matrix.mean().round(decimals=3)
    matrix = matrix.round(decimals=2)

    # Create figure and axes
    fig, ax = plt.subplots()
    ax.imshow(matrix)

    # Show ticks
    ax.set_xticks(np.arange(len(ranges)))
    ax.set_yticks(np.arange(len(classes)))

    # Label ticks
    ax.set_xticklabels(ranges)
    ax.set_yticklabels(classes)

    # Rotate tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data to create annotations.
    for i in range(len(ranges)):
        for j in range(len(classes)):
            ax.text(i, j, matrix[j, i],
                    ha="center", va="center", color="w")

    # Plot figure
    ax.set_title(title.format(result))
    fig.tight_layout()

    # Save and show
    plt.savefig('{}/summary_table.png'.format(folder))
    plt.close()


def create_bar_plot(key_range, key_class, matrix, name, idx, folder):

    # Prepare title and start plot
    title = 'Per-frame depth evaluation of **{} at {}m**'.format(key_class, key_range)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get x ticks and values
    x_ticks = [int(m[0]) for m in matrix]
    x_values = range(len(matrix))
    # Get y values
    y_values = [m[2 + idx] for m in matrix]

    # Prepare titles, ticks and labels
    ax.set_title(title)
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_ticks)
    ax.set_xlabel('Image frame')
    ax.set_ylabel('{}'.format(name.upper()))

    # Rotate tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=70, ha="right",
             rotation_mode="anchor")

    # Show and save
    ax.bar(x_values, y_values)
    plt.savefig('{}/{}-{}m-{}.png'.format(folder, key_class, key_range, name))


def load_sem_ins(file):
    """Load GT semantic and instance maps"""
    sem = file.replace('_gt', '_sem')
    if os.path.isfile(sem):
        ins = file.replace('_gt', '_ins')
        sem = cv2.imread(sem, cv2.IMREAD_ANYDEPTH) / 256.
        ins = cv2.imread(ins, cv2.IMREAD_ANYDEPTH) / 256.
    else:
        sem = ins = None
    return sem, ins


def load_depth(depth):
    """Load a depth map"""
    depth = cv2.imread(depth, cv2.IMREAD_ANYDEPTH) / 256.
    depth = torch.tensor(depth).unsqueeze(0).unsqueeze(0)
    return depth


def compute_depth_metrics(config, gt, pred, use_gt_scale=True,
                          extra_mask=None, min_num_valid_pixels=1):
    """
    Compute depth metrics from predicted and ground-truth depth maps

    Parameters
    ----------
    config : CfgNode
        Metrics parameters
    gt : torch.Tensor
        Ground-truth depth map [B,1,H,W]
    pred : torch.Tensor
        Predicted depth map [B,1,H,W]
    use_gt_scale : bool
        True if ground-truth median-scaling is to be used
    extra_mask : torch.Tensor
        Extra mask to be used for calculation (e.g. semantic mask)
    min_num_valid_pixels : int
        Minimum number of valid pixels for the image to be considered

    Returns
    -------
    metrics : torch.Tensor [7]
        Depth metrics (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3)
    """
    # Initialize variables
    batch_size, _, gt_height, gt_width = gt.shape
    abs_diff = abs_rel = sq_rel = rmse = rmse_log = silog = a1 = a2 = a3 = 0.0
    # For each depth map
    for pred_i, gt_i in zip(pred, gt):
        gt_i, pred_i = torch.squeeze(gt_i), torch.squeeze(pred_i)

        # Keep valid pixels (min/max depth and crop)
        valid = (gt_i > config.min_depth) & (gt_i < config.max_depth)
        valid = valid & torch.squeeze(extra_mask) if extra_mask is not None else valid

        # Stop if there are no remaining valid pixels
        if valid.sum() < min_num_valid_pixels:
            return None, None

        # Keep only valid pixels
        gt_i, pred_i = gt_i[valid], pred_i[valid]

        # Ground-truth median scaling if needed
        if use_gt_scale:
            pred_i = pred_i * torch.median(gt_i) / torch.median(pred_i)

        # Clamp predicted depth values to min/max values
        pred_i = pred_i.clamp(config.min_depth, config.max_depth)

        # Calculate depth metrics

        thresh = torch.max((gt_i / pred_i), (pred_i / gt_i))
        a1 += (thresh < 1.25     ).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        diff_i = gt_i - pred_i
        abs_diff += torch.mean(torch.abs(diff_i))
        abs_rel += torch.mean(torch.abs(diff_i) / gt_i)
        sq_rel += torch.mean(diff_i ** 2 / gt_i)
        rmse += torch.sqrt(torch.mean(diff_i ** 2))
        rmse_log += torch.sqrt(torch.mean((torch.log(gt_i) -
                                           torch.log(pred_i)) ** 2))

        err = torch.log(pred_i) - torch.log(gt_i)
        silog += torch.sqrt(torch.mean(err ** 2) - torch.mean(err) ** 2) * 100

    # Return average values for each metric
    return torch.tensor([metric / batch_size for metric in
        [abs_rel, sq_rel, rmse, rmse_log, silog, a1, a2, a3]]).type_as(gt), valid.sum()


def main(args):

    # Get and sort ground-truth and predicted files
    pred_files = glob(os.path.join(args.pred_folder, '*.png'))
    pred_files.sort()

    gt_files = glob(os.path.join(args.gt_folder, '*_gt.png'))
    gt_files.sort()

    depth_ranges = args.ranges
    depth_classes = args.classes

    print('#### Depth ranges to evaluate:', depth_ranges)
    print('#### Depth classes to evaluate:', depth_classes)
    print('#### Number of predicted and groundtruth files:', len(pred_files), len(gt_files))

    # Metrics name
    metric_names = ['abs_rel', 'sqr_rel', 'rmse', 'rmse_log', 'silog', 'a1', 'a2', 'a3']
    matrix_metric = 'rmse'

    # Prepare matrix information
    matrix_idx = metric_names.index(matrix_metric)
    matrix = np.zeros((len(depth_classes), len(depth_ranges)))

    # Create metrics dictionary
    all_metrics = OrderedDict()
    for depth in depth_ranges:
        all_metrics[depth] = OrderedDict()
        for classes in depth_classes:
            all_metrics[depth][classes] = []

    assert len(pred_files) == len(gt_files), 'Wrong number of files'

    # Loop over all files
    progress_bar = tqdm(zip(pred_files, gt_files), total=len(pred_files))
    for i, (pred_file, gt_file) in enumerate(progress_bar):
        # Get and prepare ground-truth and predictions
        pred = load_depth(pred_file)
        gt = load_depth(gt_file)
        pred = torch.nn.functional.interpolate(pred, gt.shape[2:], mode='nearest')
        # Check for semantics
        sem = gt_file.replace('_gt.png', '_sem.png')
        with_semantic = os.path.exists(sem)
        if with_semantic:
            sem = torch.tensor(load_sem_ins(sem)[0]).unsqueeze(0).unsqueeze(0)
            if sem.max() < 1.0:
                sem = sem * 256
            sem = torch.nn.functional.interpolate(sem, gt.shape[2:], mode='nearest')
            sem = convert_ontology(sem, ddad_to_cityscapes)
        else:
            pass
        # Calculate metrics
        for key_depth in all_metrics.keys():
            for key_class in all_metrics[key_depth].keys():
                # Prepare config dictionary
                args_key = Namespace(**{
                    'min_depth': 0,
                    'max_depth': key_depth,
                })
                # Initialize metrics as None
                metrics, num = None, None
                # Considering all pixels
                if key_class == 'All':
                    metrics, num = compute_depth_metrics(
                        args_key, gt, pred, use_gt_scale=args.use_gt_scale)
                # Considering semantic classes
                elif with_semantic:
                    metrics, num = compute_depth_metrics(
                        args_key, gt, pred, use_gt_scale=args.use_gt_scale,
                        extra_mask=sem == map_classes[key_class],
                        min_num_valid_pixels=args.min_num_valid_pixels)
                # Store metrics if available
                if metrics is not None:
                    metrics = metrics.detach().cpu().numpy()
                    metrics = np.array([i, num] + list(metrics))
                    all_metrics[key_depth][key_class].append(metrics)

    if args.output_folder is None:
        out_dict = {}
        # Loop over range values
        for key1, val1 in all_metrics.items():
            # Loop over depth metrics
            for key2, val2 in val1.items():
                key = '{}_{}m'.format(key2, key1)
                if len(val2) > 0:
                    out_dict[key] = {}
                    for i in range(len(metric_names)):
                        idx = [val2[j][0] for j in range(len(val2))]
                        nums = [val2[j][1] for j in range(len(val2))]
                        vals = [val2[j][i+2] for j in range(len(val2))]
                        out_dict[key]['{}'.format(metric_names[i])] = sum(
                            [n * v for n, v in zip(nums, vals)]) / sum(nums)
                        vals = [val2[j][i+2] for j in range(len(val2))]
                        out_dict[key]['{}'.format(metric_names[i])] = sum(vals) / len(vals)
                else:
                    out_dict[key] = None

        m_abs_rel = {}
        for key, val in out_dict.items():
            if 'All' not in key:
                m_abs_rel[key] = val['abs_rel'] if val is not None else None
        m_abs_rel = sum([val for val in m_abs_rel.values()]) / len(m_abs_rel.values())

        filtered_dict = {
            'AbsRel': out_dict['All_200m']['abs_rel'],
            'RMSE': out_dict['All_200m']['rmse'],
            'SILog': out_dict['All_200m']['silog'],
            'a1': out_dict['All_200m']['a1'],
            'Car_AbsRel': out_dict['Car_200m']['abs_rel'],
            'Person_AbsRel': out_dict['Person_200m']['abs_rel'],
            'mAbsRel': m_abs_rel,
        }

        return filtered_dict

    # Terminal lines
    met_line = '| {:>11} | {:^5} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} |'
    hor_line = '|{:<}|'.format('-' * 109)
    num_line = '| {:>10}m | {:>5} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} |'
    # File lines
    hor_line_file = '|{:<}|'.format('-' * 106)
    met_line_file = '| {:>8} | {:^5} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} | {:^8} |'
    num_line_file = '| {:>8} | {:>5} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} | {:^8.3f} |'
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)

    # Loop over the dataset
    for i, key_class in enumerate(depth_classes):
        # Create file and write header
        file = open('{}/{}.txt'.format(args.output_folder, key_class), 'w')
        file.write(hor_line_file + '\n')
        file.write('| ***** {} *****\n'.format(key_class.upper()))
        # Print header
        print(hor_line)
        print(met_line.format(*((key_class.upper()), '#') + tuple(metric_names)))
        print(hor_line)
        # Loop over each depth range and semantic class
        for j, key_depth in enumerate(depth_ranges):
            metrics = all_metrics[key_depth][key_class]
            if len(metrics) > 0:
                # How many metrics were generated for that combination
                length = len(metrics)
                # Update file
                file.write(hor_line_file + '\n')
                file.write(met_line_file.format(*('{}m'.format(key_depth), '#') + tuple(metric_names)) + '\n')
                file.write(hor_line_file + '\n')
                # Create bar plot
                create_bar_plot(key_depth, key_class, metrics, matrix_metric, matrix_idx, args.output_folder)
                # Save individual metric to file
                for metric in metrics:
                    idx, qty, metric = int(metric[0]), int(metric[1]), metric[2:]
                    file.write(num_line_file.format(*(idx, qty) + tuple(metric)) + '\n')
                # Average metrics and update matrix
                metrics = (sum(metrics) / len(metrics))
                matrix[i, j] = metrics[2 + matrix_idx]
                # Print to terminal
                print(num_line.format(*((key_depth, length) + tuple(metrics[2:]))))
                # Update file
                file.write(hor_line_file + '\n')
                file.write(num_line_file.format(*('TOTAL', length) + tuple(metrics[2:])) + '\n')
                file.write(hor_line_file + '\n')
        # Finish file
        file.write(hor_line_file + '\n')
        file.close()
    # Finish terminal printing
    print(hor_line)
    # Create final results
    create_summary_table(depth_ranges, depth_classes, matrix, args.output_folder, args.metric)


if __name__ == '__main__':
    args = parse_args()
    main(args)
