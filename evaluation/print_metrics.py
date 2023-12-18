"""Print the key metrics of multiple experiments to the standard output.
"""

__author__ = "Paul Bergmann, David Sattlegger"
__copyright__ = "2021, MVTec Software GmbH"

import argparse
import json
import os
from os.path import join

import numpy as np
from tabulate import tabulate

from generic_util import OBJECT_NAMES


def parse_user_arguments():
    """Parse user arguments.

    Returns: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="""Parse user arguments.""")

    parser.add_argument('--metrics_folder',
                        default="./metrics/",
                        help="""Path to the folder that contains the evaluation
                                results.""")

    return parser.parse_args()


def extract_table_rows(metrics_folder, metric):
    """Extract all rows to create a table that displays a given metric for each
    evaluated experiment.

    Args:
        metrics_folder: Base folder that contains evaluation results.
        metric: Name of the metric to be extracted. Choose between
          'au_pro' for localization and
          'classification_au_roc' for classification.

    Returns:
        List of table rows. Each row contains the experiment name and the
          extracted metrics for each evaluated object as well as the mean
          performance.
    """
    assert metric in ['au_pro', 'classification_au_roc']

    # Iterate each experiment.
    exp_ids = os.listdir(metrics_folder)
    exp_id_to_json_path = {
        exp_id: join(metrics_folder, exp_id, 'metrics.json')
        for exp_id in exp_ids
        if os.path.exists(join(metrics_folder, exp_id, 'metrics.json'))
    }

    # If there is a metrics.json file in the metrics_folder, also print that.
    # This is the case when evaluate_experiment.py has been called.
    root_metrics_json_path = join(metrics_folder, 'metrics.json')
    if os.path.exists(root_metrics_json_path):
        exp_id = join(os.path.split(metrics_folder)[-1], 'metrics.json')
        exp_id_to_json_path[exp_id] = root_metrics_json_path

    rows = []
    for exp_id, json_path in exp_id_to_json_path.items():

        # Each row starts with the name of the experiment.
        row = [exp_id]

        # Open the metrics file.
        with open(json_path) as file:
            metrics = json.load(file)

        # Parse performance metrics for each evaluated object if available.
        for obj in OBJECT_NAMES:
            if obj in metrics:
                row.append(np.round(metrics[obj][metric], decimals=3))
            else:
                row.append('-')

        # Parse mean performance.
        row.append(np.round(metrics['mean_' + metric], decimals=3))
        rows.append(row)

    return rows


def main():
    """Print the key metrics of multiple experiments to the standard output.
    """
    # Parse user arguments.
    args = parse_user_arguments()

    # Create the table rows. One row for each experiment.
    rows_pro = extract_table_rows(args.metrics_folder, 'au_pro')
    rows_roc = extract_table_rows(args.metrics_folder, 'classification_au_roc')

    # Print localization result table.
    print("\nAU PRO (localization)")
    print(
        tabulate(
            rows_pro, headers=['Experiment'] + OBJECT_NAMES + ['Mean'],
            tablefmt='fancy_grid'))

    # Print classification result table.
    print("\nAU ROC (classification)")
    print(
        tabulate(
            rows_roc, headers=['Experiment'] + OBJECT_NAMES + ['Mean'],
            tablefmt='fancy_grid'))


if __name__ == "__main__":
    main()
