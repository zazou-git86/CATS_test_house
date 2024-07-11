"""
Copyright (c) 2024 Orange - All rights reserved

Author:  Joël Roman Ky
This code is distributed under the terms and conditions
of the MIT License (https://opensource.org/licenses/MIT)
"""

from itertools import groupby

import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, random_split, DataLoader

from utils.datasets.cg import data_processing
#from utils.time_series_dataset import TimeSeriesDataset
from utils.cloud_gaming_dataset import CloudGamingDataset




def get_cg_data(data_dir, platform='std', window_size=10, stride=1,
                cont_rate=0, wad_threshold=.8, threshold=.8,
                is_consecutive=False):

    # Get the data
    data_dict, _ = data_processing(data_dir,
                                                    window_size=window_size,
                                                    contamination_ratio=cont_rate,
                                                    wad_threshold=wad_threshold,
                                                    stride=stride,
                                                    seed=42,
                                                    threshold=threshold,
                                                    is_consecutive=is_consecutive,
                                                    platform=platform)
    data_dict = standard_data(data_dict)

    # Get torch dataset
    train_data = CloudGamingDataset(data=data_dict['train']['x'],
                                    labels=None)
    test_data = CloudGamingDataset(data=data_dict['test']['x'],
                                    labels=data_dict['test']['y'])
    return train_data, test_data
    
def standard_data(data_dict):
    x_train = data_dict['train']['x']
    #y_train = data_dict['train']['window_labels']
    x_test = data_dict['test']['x']
    #y_test = data_dict['test']['window_labels']

    n_train, w, d = x_train.shape
    n_test, w, d = x_test.shape


    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train.reshape(n_train, w*d))
    x_test_scaled = scaler.transform(x_test.reshape(n_test, w*d))

    data_dict['train']['x'] = x_train_scaled.reshape(n_train, w, d)
    data_dict['test']['x'] = x_test_scaled.reshape(n_test, w, d)

    return data_dict

def get_train_val_data_loaders(data: Dataset,
                            batch_size: int,
                            train_ratio: float = .7,
                            shuffle: bool = True):

    train_size = int(train_ratio * len(data))
    val_size = len(data) - train_size

    # Split into train and val datasets
    train_dataset, val_dataset = random_split(data, [train_size, val_size])

    # Create the dataloaders

    train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle, num_workers=2)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle, num_workers=2)
    return train_loader, val_loader

def identify_window_anomaly(window, window_size, threshold, pos_val=1.0, is_consecutive=False):
    """_summary_

    Args:
        window (_type_): _description_
        window_size (_type_): _description_
        threshold (_type_): _description_
        pos_val (float, optional): _description_. Defaults to 1.0.
        is_consecutive (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
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
