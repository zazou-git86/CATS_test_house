"""
Copyright (c) 2024 Orange - All rights reserved

Author:  Joël Roman Ky
This code is distributed under the terms and conditions
of the MIT License (https://opensource.org/licenses/MIT)
"""

import logging
import sys
import os
import pickle

import numpy as np
#import pandas as pd
from sklearn.ensemble import IsolationForest
import torch

from utils.algorithm_utils import get_train_val_data_loaders
from utils.evaluation_utils import compute_results_csv


# Get logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))



class IForest():
    """Isolation Forest
    """
    def __init__(self, estimators=100):
        """
        Args:
            estimators (int, optional): Number of estimators. Defaults to 100.
        """
        self.estimators = estimators
        self.model = IsolationForest(n_estimators=estimators)

    def train(self, train_loader):
        """Train the model.

        Args:
            train_loader (torch.Dataloader): Train dataloader.
        """
        train_data = []
        for data in train_loader:
            ts_batch = data
            feat_size = ts_batch.shape[-1]
            ts_batch = ts_batch.reshape(-1, feat_size)
            train_data.append(ts_batch.cpu().numpy())
        train_data = np.concatenate(train_data, axis=0)
        self.model.fit(train_data)

    def test(self, test_loader):
        """Test the model

        Args:
            test_loader (torch.Dataloader): Test dataloader.

        Returns:
            (np.array*): Anomaly score and Label arrays.
        """
        test_data = []
        labels_list = []
        for data in test_loader:
            ts_batch, label = data
            feat_size = ts_batch.shape[-1]
            ts_batch = ts_batch.reshape(-1, feat_size)
            test_data.append(ts_batch.cpu().numpy())
            labels_list.append(label.cpu().numpy())
        test_data = np.concatenate(test_data, axis=0)
        labels_list = np.concatenate(labels_list, axis=0)

        anomaly_score = (-1.0)*self.model.decision_function(test_data)
        return anomaly_score, labels_list

def eval_on_data(dataset, save_dir, batch_size=512):
    """Train and evaluate the model on a dataset.

    Args:
        dataset (torch.Dataset): Dataset.
        save_dir (str): Path to the save directory.
        batch_size (int, optional): Batch size. Defaults to 512.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    out_path = os.path.join(save_dir, 'models')
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    win_size = dataset.win_size
    dataset.set_flag('train')
    train_loader, _ = get_train_val_data_loaders(dataset, batch_size=batch_size, train_ratio=.9)

    model = IForest()
    model.train(train_loader)

    dataset.set_flag('test')
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                shuffle=False)

    anomaly_score, labels_list = model.test(test_dataloader)
    anomaly_score = anomaly_score.reshape(-1, win_size)
    compute_results_csv(anomaly_score, labels_list, win_size, save_dir)

    # Save the models
    with open(out_path + '/model', "wb") as file:
        pickle.dump(model.model, file)

def load_model(save_dir):
    """Load model.

    Args:
        save_dir (str): Path to the folder where outputs are stored.

    Returns:
        sklearn.Model: Model
    """

    # init params must contain only arguments of algo_class's constructor
    model = IForest()
    #device = algo.device
    model_path = os.path.join(save_dir, 'models', 'model')
    with open(model_path, 'rb') as file:
        model.model = pickle.load(file)

    return model
