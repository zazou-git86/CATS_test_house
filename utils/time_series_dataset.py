"""
Copyright (c) 2024 Orange - All rights reserved

Author:  Joël Roman Ky
This code is distributed under the terms and conditions
of the MIT License (https://opensource.org/licenses/MIT)
"""

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

from utils.datasets.cg import data_processing
from utils.datasets.smd_entity import Smd_entity
from utils.datasets.msl_entity import Msl_entity
from utils.datasets.smap_entity import Smap_entity
from utils.datasets.entities_name import entities_dict



class TimeSeriesDataset(Dataset):
    """Time Series Dataset.
    """
    def __init__(self, data_path,
                win_size, stride,
                augmentation=None,
                data_name='gfn',
                entity=False,
                entity_number=0.,
                cont_rate=0,
                flag='train',
                pos_augment_1=None,
                pos_augment_2=None,
                neg_augment_func_list=None,
                seed=42,
                negative=True):
        """
        Args:
            data_path (str): Path to datasets.
            win_size (int): Window size.
            stride (int): Stride.
            augmentation (Augmentation, optional): Data augmentation. Defaults to None.
            data_name (str, optional): Name of the dataset. Defaults to 'gfn'.
            entity (bool, optional): Use multi-entity datasets. Defaults to False.
            entity_number (int, optional): Entity number. Defaults to 0.
            cont_rate (float, optional): Contamination rate for [std, gfn, xc]. Defaults to 0.
            flag (str, optional): Train or test mode. Defaults to 'train'.
            pos_augment_1 (list(str), optional): 1st positive augmentation. Defaults to None.
            pos_augment_2 (list(str), optional): 2nd positive augmentation. Defaults to None.
            neg_augment_func_list (list(str), optional): Negative augmentation. Defaults to None.
            seed (int, optional): Random seed. Defaults to 42.
            negative (bool, optional): Apply negative augmentation. Defaults to True.
        """

        self.data_path = data_path
        self.flag = flag
        self.stride = stride
        self.win_size = win_size
        self.scaler = StandardScaler()
        #self.scaler = MinMaxScaler()

        # Load data
        train_data, test_data, labels = get_data(self.data_path, data_name,
                                                window_size=win_size,
                                                stride=stride,
                                                seed=seed,
                                                entity=entity,
                                                cont_rate=cont_rate,
                                                entity_number=entity_number)
        if data_name in ['std', 'gfn', 'xc']:
            self.stride = win_size

        self.n_features = train_data.shape[-1]
        self.train_data = train_data
        self.test_data = test_data
        # Standardize the data
        self.scaler.fit(train_data)
        #print(train_data[0:2])
        train_data = self.scaler.transform(train_data)
        #print(train_data[0:2])
        test_data = self.scaler.transform(test_data)
        self.train = torch.tensor(train_data, dtype=torch.float32)
        self.test = torch.tensor(test_data, dtype=torch.float32)
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

        if self.flag == 'train':
            self.data = self.train
        elif self.flag == 'test':
            self.data = self.test

        self.augmentation = augmentation
        self.pos_augment_1 = pos_augment_1
        self.pos_augment_2 = pos_augment_2
        #self.pos_augment_func_list = pos_augment_func_list
        self.neg_augment_func_list = neg_augment_func_list
        #self.positive = positive
        self.negative = negative

    def set_negative_aug(self, neg_augment_func_list):
        """Modify negative augmentation.

        Args:
            neg_augment_func_list (list): Negative augmentation.
        """
        self.neg_augment_func_list = neg_augment_func_list

    def set_augmentation(self, new_aug_func, pos_augment_1=None,
                        pos_augment_2=None, neg_augment_func_list=None,
                        negative=True):
        """Modify augmentation parameters.

        Args:
            new_aug_func (Augmentation): Augmentation.
            pos_augment_1 (list, optional): Positive augmentation. Defaults to None.
            pos_augment_2 (list, optional): 2nd positive augmentation. Defaults to None.
            neg_augment_func_list (list, optional): Negative augmentation. Defaults to None.
            negative (bool, optional): Apply negative augmentation. Defaults to True.
        """
        self.augmentation = new_aug_func
        self.pos_augment_1 = pos_augment_1
        self.pos_augment_2 = pos_augment_2
        self.neg_augment_func_list = neg_augment_func_list
        self.negative = negative

    def set_flag(self, new_flag):
        """Modify train/test mode.

        Args:
            new_flag (str): train or test.
        """
        self.flag = new_flag
        if self.flag == 'train':
            self.data = self.train
        elif self.flag == 'test':
            self.data = self.test

    def set_win_size(self, new_win_size):
        """Modify window size.

        Args:
            new_win_size (int): Window size.
        """
        self.win_size = new_win_size

    def set_stride(self, new_stride):
        """Modify stride.

        Args:
            new_stride (int): Stride
        """
        self.stride = new_stride

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.stride + 1
        # elif (self.flag == 'val'):
        #     return (self.val.shape[0] - self.win_size) // self.stride + 1
        elif self.flag == 'test':
            return (self.test.shape[0] - self.win_size) // self.stride + 1

    def __getitem__(self, index):
        #sample = self.data[index]
        index = index * self.stride
        if self.flag == "train":
            sample = self.data[index:index + self.win_size]
        elif self.flag == 'test':
            sample = self.data[index:index + self.win_size]
            label = self.test_labels[index:index + self.win_size]
        #print(sample.shape)

        if self.augmentation is not None:
            # Apply data augmentation
            sample_neg_list = []
            # sample_pos_1 = self.augmentation.random_augment(sample,
            #                             aug_funct_names=self.pos_augment_func_list,
            #                             )
            sample_pos_1 = self.augmentation.random_augment(sample,
                                            aug_funct_names=self.pos_augment_1)
            sample_pos_2 = self.augmentation.random_augment(sample,
                                            aug_funct_names=self.pos_augment_2)
            if self.negative:
                #sample_neg_list = []
                # for neg_augment in self.neg_augment_func_list:
                #     sample_neg_list.append(self.augmentation.random_augment(sample,
                #                             aug_funct_names=[neg_augment],
                #                             ))
                sample_neg = self.augmentation.random_augment(sample,
                                            aug_funct_names=self.neg_augment_func_list,
                                            )
                sample_neg_list.append(sample_neg)
            if self.flag == 'test':
                return sample, sample_pos_2, sample_neg_list, label
            else:
                return sample_pos_1, sample_pos_2, sample_neg_list
        else:
            #print(sample.shape)
            #x = sample.reshape(self.win_size*self.n_features)
            x = sample
            if self.flag == 'test':
                return x, label
            return x


def get_data(data_path, data_name, entity=False, entity_number=0, window_size=10,
            stride=1, cont_rate=0, seed=42):
    """Get datasets.

    Args:
        data_path (str): Path to datasets.
        data_name (str): Dataset name.
        entity (bool, optional): Entity dataset. Defaults to False.
        entity_number (int, optional): Entity number. Defaults to 0.
        window_size (int, optional): Window size. Defaults to 10.
        stride (int, optional): Stride. Defaults to 1.
        cont_rate (float, optional): Contamination rate. Defaults to 0.
        seed (int, optional): Random seed. Defaults to 42.

    Raises:
        ValueError: Dataset name not implemented.

    Returns:
        (np.array*): train, test, test_label.
    """
    if entity:
        # Entity datasets
        if data_name == 'smd':
            datasets = [Smd_entity(seed=seed, remove_unique=False, entity=entity) \
                        for entity in entities_dict['smd']]
            if entity_number >= len(datasets):
                entity_number = 0
        elif data_name == 'msl':
            datasets = [Msl_entity(seed=seed, entity=entity, remove_unique=False) \
                        for entity in entities_dict['msl']]
            if entity_number >= len(datasets):
                entity_number = 0
        elif data_name == 'smap':
            datasets = [Smap_entity(seed=seed, entity=entity, remove_unique=False) \
                        for entity in entities_dict['smap']]
            if entity_number >= len(datasets):
                entity_number = 0
        else:
            raise ValueError(f'This dataset name {data_name} is not implemented !')

        x_train, _, x_test, y_test = datasets[entity_number].data()
        #print(type(x_test), type(y_test))
        train_data = x_train.values
        test_data = x_test.values
        test_labels = y_test.values


    else:
        if data_name in ['std', 'gfn', 'xc']:
            train_data, test_data, test_labels = get_cg_data(data_path, data_name,
                                                            window_size, stride,
                                                            cont_rate=cont_rate,
                                                            seed=seed)
        else:
            if data_name == 'swat':
                #train_data = pd.read_csv(os.path.join(data_path, 'SWaT_Dataset_Normal_v1.csv'))
                train_data = pd.read_csv(os.path.join(data_path, 'swat_train2.csv'))
                #test_data = pd.read_csv(os.path.join(data_path, 'SWaT_Dataset_Attack_v0.csv'))
                test_data = pd.read_csv(os.path.join(data_path, 'swat2.csv'))
                test_labels = test_data.values[:, -1:].reshape(-1)
                train_data = train_data.values[:, :-1]
                test_data = test_data.values[:, :-1]
            elif data_name == 'psm':
                train_data = pd.read_csv(os.path.join(data_path, 'train.csv'))
                test_data = pd.read_csv(os.path.join(data_path, 'test.csv'))
                test_labels = pd.read_csv(os.path.join(data_path, 'test_label.csv'))

                train_data = train_data.values[:, 1:]
                train_data = np.nan_to_num(train_data)
                test_data = test_data.values[:, 1:]
                test_data = np.nan_to_num(test_data)
                test_labels = test_labels.values[:, 1:].reshape(-1)
            elif data_name == 'wadi':
                train_data = np.load(os.path.join(data_path, "WADI_train.npy"))
                test_data = np.load(os.path.join(data_path, "WADI_test.npy"))
                test_labels = np.load(os.path.join(data_path, "WADI_test_label.npy"))
            elif data_name == 'nips_ts_swan':
                train_data = np.load(os.path.join(data_path, "NIPS_TS_Swan_train.npy"))
                test_data = np.load(os.path.join(data_path, "NIPS_TS_Swan_test.npy"))
                test_labels = np.load(os.path.join(data_path, "NIPS_TS_Swan_test_label.npy"))
            elif data_name == 'nips_ts_creditcard':
                train_data = np.load(os.path.join(data_path, "NIPS_TS_creditcard_train.npy"))
                test_data = np.load(os.path.join(data_path, "NIPS_TS_creditcard_test.npy"))
                test_labels = np.load(os.path.join(data_path, "NIPS_TS_creditcard_test_label.npy"))
            elif data_name == 'nips_ts_gecco':
                train_data = np.load(os.path.join(data_path, "NIPS_TS_Water_train.npy"))
                test_data = np.load(os.path.join(data_path, "NIPS_TS_Water_test.npy"))
                test_labels = np.load(os.path.join(data_path, "NIPS_TS_Water_test_label.npy"))
            else:
                raise ValueError(f'This dataset name {data_name} is not implemented !')
    return train_data, test_data, test_labels


def get_cg_data(data_path, data_name, window_size, stride, cont_rate=0, seed=42):
    """Get cloud gaming datasets.

    Args:
        data_path (str): Data path.
        data_name (str): Dataset name.
        window_size (int): Window size.
        stride (int): Stride.
        cont_rate (float, optional): Contamination rate. Defaults to 0.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        (np.array): train, test, test_label arrays.
    """
    wad_threshold = 1
    threshold = 1
    is_consecutive = False

    # Get the data
    data_dict, _ = data_processing(data_path,
                                    window_size=window_size,
                                    contamination_ratio=cont_rate,
                                    wad_threshold=wad_threshold,
                                    stride=stride,
                                    seed=seed,
                                    threshold=threshold,
                                    is_consecutive=is_consecutive,
                                    platform=data_name)
    x_train = data_dict['train']['x']
    #y_train = data_dict['train']['window_labels']
    x_test = data_dict['test']['x']
    y_test = data_dict['test']['y']

    n_train, w, d = x_train.shape
    n_test, w, d = x_test.shape

    x_train = x_train.reshape(n_train*w, d)
    x_test = x_test.reshape(n_test*w, d)
    y_test = y_test.reshape(-1)

    #print(y_test.shape, x_test.shape)

    return x_train, x_test, y_test
