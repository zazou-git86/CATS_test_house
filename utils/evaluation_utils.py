"""
Copyright (c) 2024 Orange - All rights reserved

Author:  Joël Roman Ky
This code is distributed under the terms and conditions
of the MIT License (https://opensource.org/licenses/MIT)
"""

from itertools import groupby
import time
import os
import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix

def compute_auc_score(anomaly_score, labels, pos_label=1):
    """Compute AUC score.

    Args:
        anomaly_score (np.array): Anomaly score.
        labels (np.array): Labels.
        pos_label (int, optional): Positive label. Defaults to 1.

    Returns:
        float: AUC score.
    """

    # AUC scores
    fpr, tpr, _ = metrics.roc_curve(labels, anomaly_score,
                                    pos_label=pos_label)
    auc = metrics.auc(fpr, tpr)

    # AUPR score
    aupr = metrics.average_precision_score(labels, anomaly_score,
                                            pos_label=pos_label)
    return auc, aupr, fpr, tpr

def compute_report_score(labels, preds):
    """Compute F1, MCC, ACC, BAC, precision and recall.

    Args:
        labels (np.array): Labels arrays.
        preds (np.array): Prediction arrays.

    Returns:
        (float*): F1, MCC, ACC, BAC, precision and recall.
    """
    _, fp, fn, tp = metrics.confusion_matrix(labels, preds).ravel()
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    bac = metrics.balanced_accuracy_score(labels, preds)
    acc = metrics.accuracy_score(labels, preds)
    mcc = metrics.matthews_corrcoef(labels, preds)
    f1_score = metrics.f1_score(labels, preds)
    return f1_score, mcc, acc, bac, precision, recall

def compute_performance_metrics(anomaly_score, labels, threshold,
                                pos_label=1,
                                more_stats=False):
    """Compute all the scores.

    Args:
        anomaly_score (np.array): Anomaly score arrays.
        labels (np.array): Labels arrays.
        threshold (float): Anomaly threshold.
        pos_label (int, optional): Positive label. Defaults to 1.
        more_stats (bool, optional): Print all metrics. Defaults to False.

    Returns:
        dict: Performance metrics.
    """
    preds = (anomaly_score >= threshold).astype(int)

    auc, aupr, _, _ = compute_auc_score(anomaly_score,
                                            labels,
                                            pos_label=pos_label)
    f1_score, mcc, acc, bac, precision, recall = compute_report_score(labels, preds)

    base_report = {
        'AUPR': aupr,
        'AUC': auc,
        'F1': f1_score,
        'MCC': mcc,
        'BAC': bac,
    }
    add_report = {
        'precision': precision,
        'recall': recall,
        'accuracy': acc,
        # 'fpr': fpr,
        # 'tpr': tpr
    }
    if more_stats:
        return base_report | add_report
    return base_report

def get_best_f1_threshold(test_score, y_true, number_pertiles,
                            verbose=True,):
    """Get the threshold that report the best f1 score.

    Args:
        test_score (np.ndarray): Test score arrays.
        y_true (np.ndarray): Ground-truths arrays.
        y_win_adjust (np.ndarray): Window-adjust predictions.
        number_pertiles (int): Number of pertiles
        verbose (bool, optional): If print runs information. Defaults to True.
        metric_type (str, optional): Point-wise or Window metrics. Defaults to 'pw'.
        window_type (str, optional): Window type. Defaults to 'wad'.

    Raises:
        ValueError: Metric type unknown.

    Returns:
        float: Threshold.
    """
    #print(y_true.shape)
    ratio = float(100 * sum(y_true == 0) / len(y_true))
    #print(ratio, type(ratio))
    print(f"Ratio of normal data: {ratio:.2f}%")
    pertiles_list = np.linspace(max(ratio - 5, 0), min(ratio + 5, 100), number_pertiles)
    thresholds = np.percentile(test_score, pertiles_list)

    f1_list = np.zeros(shape=number_pertiles)
    mcc_list = np.zeros(shape=number_pertiles)
    recall_list = np.zeros(shape=number_pertiles)
    precision_list = np.zeros(shape=number_pertiles)

    st_tm = time.time()
    for i, (thresh, _) in enumerate(zip(thresholds, pertiles_list)):
        y_pred = (test_score >= thresh).astype(int)
        _, false_pos, false_neg, true_pos = confusion_matrix(y_true, y_pred).ravel()


        precision_list[i] = true_pos/(true_pos+false_pos)
        recall_list[i] = true_pos /(true_pos+false_neg)
        #f1_list[i] = 2*precision_list[i]*recall_list[i] / (precision_list[i] + recall_list[i])
        f1_list[i] = metrics.f1_score(y_true, y_pred)
        mcc_list[i] = metrics.matthews_corrcoef(y_true, y_pred)

    print(f'Threshold optimization took : {time.time() - st_tm:.2f} s')
    arm = np.argmax(mcc_list)
    if verbose:
        print(f"Best metrics with threshold = {thresholds[arm]:.2f} are \
            :\tPrecision = {precision_list[arm]:.2f}\tRecall = {recall_list[arm]:.2f} \
            \tF1-Score = {f1_list[arm]:.2f} \
            \tMCC = {mcc_list[arm]:.2f}\n")
    return thresholds[arm]

def get_best_score(test_score, y_true, val_ratio, n_pertiles,
                    seed=42):
    """Compute the best score based on the best threshold.

    Args:
        test_score (np.ndarray): Test scores arrays.
        y_true (np.ndarray): Ground-truths arrays.
        y_win_true (np.ndarray): Ground-truths windows arrays.
        y_win_adjust (np.ndarray): Window-adjusted arrays.
        val_ratio (float): Ratio of test set to use for best threshold determination.
        n_pertiles (int): Number of pertiles.
        metric_type (str, optional): Point-wise or window metrics. Defaults to 'pw'.
        seed (int, optional): Random seed value. Defaults to 42.
        window_type (str, optional): Window type. Defaults to 'wad'.

    Raises:
        ValueError: Metric type unknown !

    Returns:
        tuple: Score values.
    """
    #print(test_score.shape, y_true.shape, y_win_true.shape)
    # Take val_ratio % of the predictions to find the threshold that yield to the
    # best F1 score with a stratified sampling
    # Reshape ground truths from (n_samples, win_size) -> (n_samples*win_size)
    #y_true = y_true.reshape(test_score.shape[0], -1)

    # Stratified shuffle based on anomalous observations
    split = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    for _, v_index in split.split(test_score, y_true):
        #score_t, y_t = test_score[t_index], y_true[t_index]
        score_v, y_v = test_score[v_index], y_true[v_index]
    threshold = get_best_f1_threshold(score_v, y_v,
                                        number_pertiles=n_pertiles,)

    return threshold

def anomaly_plot(anomaly_score, win_label_list, log=True, save_dir=None):
    """Plot anomaly scores.

    Args:
        anomaly_score (np.array): Anomaly score.
        win_label_list (np.array): Labels arrays.
        log (bool, optional): Log-plot. Defaults to True.
        save_dir (str, optional): Path to save plots. Defaults to None.
    """
    if log:
        anomaly_score_cop = anomaly_score.copy()
        min_val = anomaly_score_cop.min()
        if min_val <0:
            anomaly_score_cop = np.add(anomaly_score_cop, -min_val)
        anomaly_score_cop[anomaly_score_cop==0] = 1e-10
        log_anomaly_score = np.log(anomaly_score_cop)
        # Check for non-finite values
        #if np.any(~np.isfinite(log_anomaly_score)):
        # Remove non-finite values
        # max_non_inf = np.nanmax(log_anomaly_score[~np.isinf(log_anomaly_score)])
        # print(max_non_inf)
        # log_anomaly_score[np.isinf(log_anomaly_score)] = max_non_inf
    else:
        log_anomaly_score = np.copy(anomaly_score)

    # Plot the anomaly scores with true labels
    hist_anomaly = log_anomaly_score[win_label_list == 1]
    #mu_ano, std_ano = norm.fit(hist_anomaly)
    hist_normal = log_anomaly_score[win_label_list == 0]
    #mu_norm, std_norm = norm.fit(hist_normal)


    plt.hist(hist_normal, bins=20, label='Normal', density=True, alpha=.5)
    plt.hist(hist_anomaly, bins=20, label='Anomaly', density=True, alpha=.5)

    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title('Anomaly Score Distribution for Normal and Anomaly Points')
    plt.legend()
    if save_dir is not None:
        plt.savefig(save_dir)
        plt.close()
    else:
        plt.show()

def identify_window_anomaly(window, window_size, threshold, pos_val=1.0, is_consecutive=False):
    """Window adjustement.

    Args:
        window (_type_): _description_
        window_size (_type_): _description_
        threshold (_type_): _description_
        pos_val (float, optional): _description_. Defaults to 1.0.
        is_consecutive (bool, optional): _description_. Defaults to False.

    Returns:
        int: Adjustement.
    """
    # print(threshold, window_size)
    if threshold > window_size:
        raise ValueError("The threshold value is above the window size")
    if is_consecutive:
        count_consec_anomaly = [(val, sum(1 for _ in group)) for val, group in
                                groupby(window) if val==pos_val]
        count_val = [tup_count[1] for tup_count in count_consec_anomaly]

    #print(count_consec_anomaly)
        if not count_val:
            total_anomal_obs, _ = 0, 0
        else:
            total_anomal_obs, _ = sum(count_val), max(count_val)
    else:
        total_anomal_obs = np.count_nonzero(window == pos_val)
    #return total_anomal_obs, max_consec_anomal
    #print(total_anomal_obs, window_size//2, max_consec_anomal, threshold)
    # an = []
    #for v in count_val:
    if total_anomal_obs >= threshold:  #(v >= threshold): # or
            #an.append(1)
        return 1
    return 0

def get_window_anomaly(labels_list, window_size=10, threshold=10, pos_val=1.0):
    """Compute adjustement for all labels.

    Args:
        labels_list (np.array): Labels arrays.
        window_size (int, optional): Window size. Defaults to 10.
        threshold (int, optional): Adjustement threshold. Defaults to 10.
        pos_val (float, optional): Positive label. Defaults to 1.0.

    Returns:
        np.array: Adjusted labels.
    """
    win_label_list = [identify_window_anomaly(labels_list[i],
                                            window_size=window_size,
                                            threshold=threshold,
                                            pos_val=pos_val,
                                            is_consecutive=False)
                    for i in range(len(labels_list))]
    #win_label_list = [label[0] for label in labels_list]
    win_label_list = np.array(win_label_list)
    return win_label_list

def compute_results_csv(anomaly_score, labels_list, win_size, save_dir, win=False):
    """Compute performance metrics and save in csv file.

    Args:
        anomaly_score (np.array): Anomaly score.
        labels_list (np.array): Labels arrays.
        win_size (int): Window size.
        save_dir (str): Path to save results.
        win (bool, optional): Apply window adjustement. Defaults to False.
    """
    if win:
        win_label_list = get_window_anomaly(labels_list,
                                            window_size=win_size,
                                            threshold=win_size,
                                            pos_val=1.0)

        threshold = get_best_score(anomaly_score.mean(axis=-1), win_label_list,
                                val_ratio=.2, n_pertiles=100, seed=42)
        stats_dict = compute_performance_metrics(anomaly_score.mean(axis=-1),
                                                win_label_list,
                                                threshold,
                                                pos_label=1,
                                                more_stats=True)

        # Save the dataframe
        result_path = os.path.join(save_dir, 'results.csv')
        pd.DataFrame([stats_dict]).to_csv(result_path, index=False)

    anomaly_score_pw = anomaly_score.reshape(-1)
    labels_pw = labels_list.reshape(-1)
    threshold_pw = get_best_score(anomaly_score_pw, labels_pw,
                            val_ratio=.2, n_pertiles=100, seed=42)

    stats_dict_pw = compute_performance_metrics(anomaly_score_pw, labels_pw, threshold_pw,
                                            pos_label=1,
                                            more_stats=True)
    # Save the dataframe
    result_pw_path = os.path.join(save_dir, 'results_pw.csv')
    pd.DataFrame([stats_dict_pw]).to_csv(result_pw_path, index=False)

    if win:
        plot_save_path = os.path.join(save_dir, 'anomaly.png')
        anomaly_plot(anomaly_score.mean(axis=-1), win_label_list, log=True, save_dir=plot_save_path)

    plot_save_path = os.path.join(save_dir, 'anomaly_pw.png')
    anomaly_plot(anomaly_score_pw, labels_pw, log=True, save_dir=plot_save_path)

def aggregate_results_over_runs(result_path, n_runs, res_type='pw'):
    """Aggregate results over many runs.

    Args:
        result_path (str): Path where the results are stored.
        n_runs (int): Number of runs.
        res_type (str, optional): Window adjustment or not. Defaults to 'pw'.

    Returns:
        _type_: _description_
    """
    df_list = []
    if res_type == 'win':
        filename = 'results.csv'
    else:
        filename = 'results_pw.csv'
    for i in range(n_runs):
        df_path = os.path.join(result_path, f'model_{i+1}', filename)
        df = pd.read_csv(df_path)
        df_list.append(df)

    df_whole = pd.concat(df_list, axis=0)

    # Calculate mean, standard deviation, and confidence intervals
    stats = df_whole.agg(['mean', 'std'])
    n = len(df_whole)

    # Compute 95% confidence intervals
    ci95_hi = []
    ci95_lo = []

    for col in df_whole.columns:
        m, s = stats.loc['mean', col], stats.loc['std', col]
        ci95_hi.append(m + 1.96 * s / math.sqrt(n))
        ci95_lo.append(m - 1.96 * s / math.sqrt(n))

    # Create a new DataFrame to store the results
    result_df = pd.DataFrame({
        'mean': stats.loc['mean'],
        'std': stats.loc['std'],
        'ci_95_low': ci95_lo,
        'ci_95_high': ci95_hi
    })
    save_path = os.path.join(result_path, filename)
    result_df.to_csv(save_path, index=True)
    return result_df
