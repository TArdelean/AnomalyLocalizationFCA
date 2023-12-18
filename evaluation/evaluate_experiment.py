"""Compute evaluation metrics for a single experiment."""

__author__ = "Paul Bergmann, David Sattlegger"
__copyright__ = "2021, MVTec Software GmbH"

import argparse
import json
from os import makedirs, path
from pathlib import Path

import numpy as np
import skimage
import skimage.transform
from PIL import Image
from tqdm import tqdm

from evaluation import generic_util as util
from evaluation.additional_util import compute_optimal_f1, compute_seg_au_roc
from evaluation.pro_curve_util import compute_pro
from evaluation.roc_curve_util import compute_classification_roc


def parse_user_arguments():
    """Parse user arguments for the evaluation of a method on the MVTec AD
    dataset.

    Returns:
        Parsed user arguments.
    """
    parser = argparse.ArgumentParser(description="""Parse user arguments.""")

    parser.add_argument('--anomaly_maps_dir',
                        required=True,
                        help="""Path to the directory that contains the anomaly
                                maps of the evaluated method.""")

    parser.add_argument('--dataset_base_dir',
                        required=True,
                        help="""Path to the directory that contains the dataset
                                images of the MVTec AD dataset.""")

    parser.add_argument('--output_dir',
                        help="""Path to the directory to store evaluation
                                results. If no output directory is specified,
                                the results are not written to drive.""")

    parser.add_argument('--padding',
                        type=int,
                        default=100,
                        help="""How much to cut of the borders of the image before evaluating;
                                the metrics are computed on m[padding:-padding, padding:-padding]""")

    parser.add_argument('--resize',
                        type=int,
                        default=0,
                        help="""Resize the predictions and ground truth masks before evaluation""")

    parser.add_argument('--pro_integration_limit',
                        type=float,
                        default=0.3,
                        help="""Integration limit to compute the area under
                                the PRO curve. Must lie within the interval
                                of (0.0, 1.0].""")

    parser.add_argument('--evaluated_objects',
                        nargs='+',
                        help="""List of objects to be evaluated. By default,
                                all dataset objects will be evaluated.""",
                        choices=util.OBJECT_NAMES,
                        default=util.OBJECT_NAMES)

    args = parser.parse_args()

    # Check that the PRO integration limit is within the valid range.
    assert 0.0 < args.pro_integration_limit <= 1.0

    return args


def parse_dataset_files(object_name, dataset_base_dir, anomaly_maps_dir):
    """Parse the filenames for one object of the MVTec AD dataset.

    Args:
        object_name: Name of the dataset object.
        dataset_base_dir: Base directory of the MVTec AD dataset.
        anomaly_maps_dir: Base directory where anomaly maps are located.
    """

    # Store a list of all ground truth filenames.
    gt_filenames = []

    # Store a list of all corresponding anomaly map filenames.
    prediction_filenames = []

    # Test images are located here.
    test_dir = path.join(dataset_base_dir, object_name, 'test')
    gt_base_dir = path.join(dataset_base_dir, object_name, 'ground_truth')
    anomaly_maps_base_dir = path.join(anomaly_maps_dir, object_name, 'test')

    # List all ground truth and corresponding anomaly images.
    for subdir in Path(test_dir).iterdir():

        # Get paths to all test images in the dataset for this subdir.
        test_images = list(subdir.glob('*.png')) + list(subdir.glob('*.jpg'))
        test_images = sorted([x.stem for x in test_images])

        # If subdir is not 'good', derive corresponding GT names.
        if subdir.name != 'good':
            file_names = (Path(gt_base_dir) / subdir.name).glob('*.png')
            gt_filenames.extend(sorted(file_names))
        else:
            # No ground truth maps exist for anomaly-free images.
            gt_filenames.extend([None] * len(test_images))

        # Fetch corresponding anomaly maps.
        prediction_filenames.extend(
            [path.join(anomaly_maps_base_dir, subdir.name, file)
             for file in test_images])

    print(f"Parsed {len(gt_filenames)} ground truth image files.")

    return gt_filenames, prediction_filenames


def calculate_au_pro_au_roc(gt_filenames,
                            prediction_filenames,
                            integration_limit,
                            padding=0,
                            resize=False):
    """Compute the area under the PRO curve for a set of ground truth images
    and corresponding anomaly images.

    In addition, the function computes the area under the ROC curve for image
    level classification.

    Args:
        gt_filenames: List of filenames that contain the ground truth images
          for a single dataset object.
        prediction_filenames: List of filenames that contain the corresponding
          anomaly images for each ground truth image.
        integration_limit: Integration limit to use when computing the area
          under the PRO curve.
        padding: Remove borders before evaluation
        resize: Resize images before padding
    Returns:
        au_pro: Area under the PRO curve computed up to the given integration
          limit.
        au_roc: Area under the ROC curve.
        pro_curve: PRO curve values for localization (fpr,pro).
        roc_curve: ROC curve values for image level classifiction (fpr,tpr).
    """
    # Read all ground truth and anomaly images.
    ground_truth = []
    predictions = []

    print("Read ground truth files and corresponding predictions...")
    original_size = np.asarray(Image.open(next(_ for _ in gt_filenames if _ is not None))).shape[-2:]
    for (gt_name, pred_name) in tqdm(zip(gt_filenames, prediction_filenames),
                                     total=len(gt_filenames)):
        prediction = util.read_tiff(pred_name)
        if resize > 0:
            prediction = skimage.transform.resize(prediction, (resize, resize))
        elif prediction.shape != original_size:
            prediction = skimage.transform.resize(prediction, original_size)
        if padding != 0:
            prediction = prediction[padding:-padding, padding:-padding]
        predictions.append(prediction)

        if gt_name is not None:
            gt = np.asarray(Image.open(gt_name))
            if resize > 0:
                gt = skimage.transform.resize(gt, (resize, resize))
            if padding != 0:
                gt = gt[padding:-padding, padding:-padding]
        else:
            gt = np.zeros(prediction.shape)
        ground_truth.append(gt)

    # Compute the PRO curve.
    pro_curve = compute_pro(
        anomaly_maps=predictions,
        ground_truth_maps=ground_truth)

    # Compute the area under the PRO curve.
    au_pro = util.trapezoid(
        pro_curve[0], pro_curve[1], x_max=integration_limit)
    au_pro /= integration_limit
    print(f"AU-PRO (FPR limit: {integration_limit}): {au_pro}")

    # Compute the segmentation ROC
    seg_roc_curve = compute_seg_au_roc(anomaly_maps=predictions, ground_truth_maps=ground_truth)
    au_seg_roc = util.trapezoid(seg_roc_curve[0], seg_roc_curve[1], x_max=1.0)
    print(f"AU-ROC segmentation: {au_seg_roc}")

    f1_optimal = compute_optimal_f1(anomaly_maps=predictions, ground_truth_maps=ground_truth)

    # Derive binary labels for each input image:
    # (0 = anomaly free, 1 = anomalous).
    binary_labels = [int(np.any(x > 0)) for x in ground_truth]
    del ground_truth

    if all([x == 1 for x in binary_labels]):
        roc_curve, au_roc = [], 1.0
    else:
        # Compute the classification ROC curve.
        roc_curve = compute_classification_roc(
            anomaly_maps=predictions,
            scoring_function=np.max,
            ground_truth_labels=binary_labels)

        # Compute the area under the classification ROC curve.
        au_roc = util.trapezoid(roc_curve[0], roc_curve[1])
        print(f"Image-level classification AU-ROC: {au_roc}")

    # Return the evaluation metrics.
    return au_pro, au_roc, pro_curve, roc_curve, seg_roc_curve, au_seg_roc, f1_optimal


def main(args):
    """Calculate the performance metrics for a single experiment on the
    MVTec AD dataset.
    """
    # Parse user arguments.
    padding = args.padding
    resize = args.resize

    # Store evaluation results in this dictionary.
    evaluation_dict = dict()

    # Keep track of the mean performance measures.
    au_pros = []
    au_rocs = []
    au_segs = []
    f1s = []

    # Evaluate each dataset object separately.
    for obj in args.evaluated_objects:
        print(f"=== Evaluate {obj} ===")
        evaluation_dict[obj] = dict()

        # Parse the filenames of all ground truth and corresponding anomaly
        # images for this object.
        gt_filenames, prediction_filenames = \
            parse_dataset_files(
                object_name=obj,
                dataset_base_dir=args.dataset_base_dir,
                anomaly_maps_dir=args.anomaly_maps_dir)

        # Calculate the PRO and ROC curves.
        au_pro, au_roc, pro_curve, roc_curve, seg_curve, au_seg, f1_optimal = \
            calculate_au_pro_au_roc(
                gt_filenames,
                prediction_filenames,
                args.pro_integration_limit, padding=padding, resize=resize)

        evaluation_dict[obj]['au_pro'] = au_pro
        evaluation_dict[obj]['classification_au_roc'] = au_roc
        evaluation_dict[obj]['au_segroc'] = au_seg
        evaluation_dict[obj]['f1_optimal_value'] = f1_optimal[0]
        evaluation_dict[obj]['f1_optimal_threshold'] = f1_optimal[1]

        # evaluation_dict[obj]['classification_roc_curve_fpr'] = roc_curve[0]
        # evaluation_dict[obj]['classification_roc_curve_tpr'] = roc_curve[1]

        # Keep track of the mean performance measures.
        au_pros.append(au_pro)
        au_rocs.append(au_roc)
        au_segs.append(au_seg)
        f1s.append(f1_optimal[0])

        print('\n')

    # Compute the mean of the performance measures.
    evaluation_dict['mean_au_pro'] = np.mean(au_pros).item()
    evaluation_dict['mean_au_segroc'] = np.mean(au_segs).item()
    evaluation_dict['mean_classification_au_roc'] = np.mean(au_rocs).item()
    evaluation_dict['mean_f1'] = np.mean(f1s).item()

    # If required, write evaluation metrics to drive.
    if args.output_dir is not None:
        output_dir = args.output_dir
        makedirs(output_dir, exist_ok=True)

        with open(path.join(output_dir, f'metrics_{padding}.json'), 'w') as file:
            json.dump(evaluation_dict, file, indent=4)

        print(f"Wrote metrics to {path.join(output_dir, f'metrics_{padding}.json')}")


if __name__ == "__main__":
    main(parse_user_arguments())
