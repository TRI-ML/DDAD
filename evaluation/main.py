# Copyright 2021 Toyota Research Institute.  All rights reserved.
#
# Validation groundtruth:
# https://tri-ml-public.s3.amazonaws.com/github/DDAD/challenge/gt_val.zip
#
# Example of validation predictions (semi-supervised):
# https://tri-ml-public.s3.amazonaws.com/github/DDAD/challenge/pred_val_sup.zip
#
# Predictions are stored as .png files in the same order as provided by the corresponding split (validation or test)
# For more information, please check our depth estimation repository: https://github.com/tri-ml/packnet-sfm
#
# How to run:
# python3 main.py gt_val.zip pred_val_sup.zip semi


import sys
import shutil

from argparse import Namespace
from zipfile import ZipFile

from semantic_evaluation import main as SemanticEval


def evaluate(gt_zip, pred_zip, phase):

    assert phase in ['semi', 'self'], 'Invalid phase name'

    use_gt_scale = phase == 'self'

    gt_folder = 'data/gt'
    print('gt_zip:', gt_zip)
    print('gt_folder:', gt_folder)
    with ZipFile(gt_zip, 'r') as zip:
        shutil.rmtree(gt_folder, ignore_errors=True)
        zip.extractall(path=gt_folder)
    pred_folder = 'data/pred'
    print('pred_zip:', pred_zip)
    print('pred_folder:', pred_folder)
    with ZipFile(pred_zip, 'r') as zip:
        shutil.rmtree(pred_folder, ignore_errors=True)
        zip.extractall(path=pred_folder)

    ranges = [200]
    metric = 'abs_rel'

    classes = [
        "All",
        "Road",
        "Sidewalk",
        "Wall",
        "Fence",
        "Building",
        "Pole",
        "T.Light",
        "T.Sign",
        "Vegetation",
        "Terrain",
        "Person",
        "Rider",
        "Car",
        "Truck",
        "Bus",
        "Bicycle",
    ]

    args = Namespace(**{
        'gt_folder': gt_folder,
        'pred_folder': pred_folder,
        'ranges': ranges, 'classes': classes,
        'metric': metric,
        'output_folder': None,
        'min_num_valid_pixels': 1,
        'use_gt_scale': use_gt_scale,
    })

    dict_output = SemanticEval(args)
    print(dict_output)

if __name__ == "__main__":
    gt_zip = sys.argv[1]    # Groundtruth .zip folder
    pred_zip = sys.argv[2]  # Predicted .zip folder
    phase = sys.argv[3]     # Which phase will be used ('semi' or 'self')
    evaluate(gt_zip, pred_zip, phase)

