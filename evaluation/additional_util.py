from sklearn import metrics
import numpy as np


def compute_seg_au_roc(anomaly_maps, ground_truth_maps):
    anomaly_scores_flat = np.array(anomaly_maps).ravel()
    gt_flat = np.array(ground_truth_maps).ravel() > 0
    fpr, tpr, thresholds = metrics.roc_curve(gt_flat.astype(int), anomaly_scores_flat)
    # au_roc = metrics.roc_auc_score(gt_flat.astype(int), anomaly_scores_flat)
    return fpr, tpr


def compute_precision_recall(anomaly_maps, ground_truth_maps):
    anomaly_scores_flat = np.array(anomaly_maps).ravel()
    gt_flat = np.array(ground_truth_maps).ravel() > 0
    precision, recall, thresholds = metrics.precision_recall_curve(gt_flat.astype(int), anomaly_scores_flat)
    return precision, recall, thresholds


def compute_optimal_f1(anomaly_maps, ground_truth_maps):
    precision, recall, thresholds = compute_precision_recall(anomaly_maps, ground_truth_maps)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    ind = np.argmax(f1_score)
    return f1_score[ind].item(), thresholds[ind].item()
