"""
Copyright (c) 2024 Orange - All rights reserved

Author:  Joël Roman Ky
This code is distributed under the terms and conditions
of the MIT License (https://opensource.org/licenses/MIT)
"""

import random

import numpy as np
import torch


class TimeSeriesAugmentation:
    """Time Series Data Augmentation
    """
    def __init__(self, spike_factor=3,
                jitter_factor=0.05,
                reduce_ratio=.9,
                window_ratio=.1,
                scales=None,
                scales_ratio=None,
                drift_slope=0.05,
                mask_ratio=None,
                seed=42,
                min_dim=.2,
                max_dim=.9,
                ):
        """

        Args:
            spike_factor (int, optional): Spike magnitude. Defaults to 3.
            jitter_factor (float, optional): Jitter standard deviation. Defaults to 0.05.
            reduce_ratio (float, optional): _description_. Defaults to .9.
            window_ratio (float, optional): _description_. Defaults to .1.
            scales (list, optional): List of [min_scale, max_scale] magnitude factor.
                                    Defaults to [.5, 2.].
            scales_ratio (list, optional): List of [min_scale, max_scale] ratio.
                                    Defaults to  [.1, .5].
            drift_slope (float, optional): Trend drift slope. Defaults to 0.05.
            mask_ratio (list, optional): List of [min_mask, max_mask] ratio.
                                    Defaults to [.1, .5].
            seed (int, optional): Random seed. Defaults to 42.
            min_dim (float, optional): Min ratio of features to augment. Defaults to .2.
            max_dim (float, optional): Max ratio of features to augment. Defaults to .9.
        """
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.spike_factor = spike_factor
        self.jitter_factor = jitter_factor

        self.reduce_ratio = reduce_ratio

        self.drift_slope = drift_slope

        self.mask_ratio = mask_ratio if mask_ratio is not None else [.1, .5]

        self.window_ratio = window_ratio

        self.scales = scales if scales is not None else [.5, 2.]
        self.scales_ratio = scales_ratio if scales_ratio is not None else [.1, .5]

        self.positives = {
                        'jitter': self.jitter,
                        'window_slicing' : self.window_slicing,
                        'window_warping' : self.window_warping,
                        'none': self.none_augment,
                        }
        self.negatives = {'spike': self.extreme_spike,
                            'trend': self.trend,
                            'shuffle': self.shuffle,
                            'scaling': self.scaling,
                            'masking': self.random_mask,
                        }
        self.seed = seed

    def extreme_spike(self, time_series, spike_ratio=.1):
        """Spike augmentation.

        Args:
            time_series (torch.Tensor): Window time series.
            spike_ratio (float, optional): Spike magnitude ratio. Defaults to .1.

        Returns:
            torch.Tensor: Augmented time series.
        """
        #spikes_num = min(spike_ratio, time_series.shape[0])
        #num_spikes = random.randint(1, spikes_num) # Randomly choose the number of spikes
        num_spikes = int(spike_ratio*time_series.shape[0])
        np.random.seed(self.seed)
        shuffled_indices = np.random.permutation(time_series.shape[0])
        indices = shuffled_indices[:num_spikes]

        dim_ratio = random.uniform(self.min_dim, self.max_dim)
        select_dim = int(dim_ratio * time_series.shape[-1])
        selected_indices = np.random.choice(time_series.shape[-1], select_dim, replace=False)

        for i in range(time_series.shape[-1]):
            if i in selected_indices:
                max_feat = time_series[:, i].max()

                spike_value = random.uniform(max_feat, max_feat * self.spike_factor)
                time_series[indices, i] = spike_value
        return time_series

    def trend(self, time_series):
        """Trebd augmentation.

        Args:
            time_series (torch.Tensor): Window time series.

        Returns:
            torch.Tensor: Augmented time series.
        """
        # Generate a linear drift component
        sequence_length, n_features = time_series.size()
        # Create a time vector
        time = torch.arange(sequence_length, dtype=torch.float32).unsqueeze(1)

        drift_factor = random.uniform(0.01, self.drift_slope)
        drift = time * drift_factor  # Linear drift component
        drift = drift.expand(sequence_length, n_features)

        dim_ratio = random.uniform(self.min_dim, self.max_dim)
        select_dim = int(dim_ratio * time_series.shape[-1])
        selected_indices = np.random.choice(time_series.shape[-1], select_dim, replace=False)

        # Add the drift to the time series
        time_series[:, selected_indices] += drift[:, selected_indices]
        return time_series

    def shuffle(self, time_series, max_segments=5, seg_mode="equal"):
        """Shuffle augmentation.

        Args:
            time_series (torch.Tensor): Window time series.
            max_segments (int, optional): Number of segments when dividing the time series.
                                            Defaults to 5.
            seg_mode (str, optional): Segments of equal size or random size.
                                            Defaults to "equal".

        Returns:
            torch.Tensor: Augmented time series.
        """
        orig_steps = np.arange(time_series.shape[0])
        num_segs = np.random.randint(3, max_segments)

        #im_ratio = random.uniform(self.min_dim, self.max_dim)
        dim_ratio = random.uniform(self.min_dim, self.max_dim)
        select_dim = int(dim_ratio * time_series.shape[-1])
        selected_indices = np.random.choice(time_series.shape[-1], select_dim, replace=False)

        if num_segs > 1:
            if seg_mode == "random":
                split_points = np.random.choice(time_series.shape[0]-2, num_segs-1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs)
            #warp = np.concatenate(np.random.permutation(splits)).ravel()
            random.shuffle(splits) # inplace shuffle
            warp = np.concatenate(splits).ravel()
            ret = time_series.clone().detach()
            for i in selected_indices:
                ret[:, i] = time_series[warp, i].clone().detach()
        else:
            ret = time_series.clone().detach()
        # print(type(ret))

        return ret

    def jitter(self, time_series):
        """Jitter augmentation.

        Args:
            time_series (torch.Tensor): Window time series.

        Returns:
            torch.Tensor: Augmented time series.
        """
        dim_ratio = random.uniform(self.min_dim, self.max_dim)
        select_dim = int(dim_ratio * time_series.shape[-1])
        selected_indices = np.random.choice(time_series.shape[-1], select_dim, replace=False)

        jitter = torch.randn_like(time_series[:, selected_indices]) * self.jitter_factor
        time_series[:, selected_indices] = time_series[:, selected_indices] + jitter
        return time_series

    def scaling(self, time_series):
        """Scaling augmentation.

        Args:
            time_series (torch.Tensor): Window time series.

        Returns:
            torch.Tensor: Augmented time series.
        """
        # Randomly choose the scaling factor
        scaling_factor = random.uniform(self.scales[0], self.scales[1])
        #scaling_factor = self.scales[1]

        # Randomly choose scale_size-consecutive timesteps
        scale_ratio = random.uniform(self.scales_ratio[0], self.scales_ratio[1])
        #scale_ratio = self.scales_ratio[1]
        scale_size = int(scale_ratio * time_series.size(0))

        # Scale the selected area
        max_start_index = time_series.size(0) - scale_size + 1
        start_index = np.random.randint(0, max_start_index)

        # Select the dimension to scales
        dim_ratio = random.uniform(self.min_dim, self.max_dim)
        select_dim = int(dim_ratio * time_series.shape[-1])
        selected_indices = np.random.choice(time_series.shape[-1], select_dim, replace=False)

        time_series[start_index : start_index + scale_size, selected_indices] *= scaling_factor
        return time_series

    def none_augment(self, time_series):
        """None augmentation.

        Args:
            time_series (torch.Tensor): Window time series.

        Returns:
            torch.Tensor: Augmented time series.
        """
        return time_series

    def random_mask(self, time_series):
        """Masking augmentation.

        Args:
            time_series (torch.Tensor): Window time series.

        Returns:
            torch.Tensor: Augmented time series.
        """
        n_samples = time_series.size(0)
        mask_factor = random.uniform(self.mask_ratio[0], self.mask_ratio[1])
        #mask_factor = self.mask_ratio[1]
        n_masked = int(n_samples * mask_factor)
        mask = torch.zeros(n_samples)
        mask[:n_masked] = 1

        mask_idx = torch.randperm(n_samples)
        mask = mask[mask_idx] == 1

        # Select the dimension to scales
        dim_ratio = random.uniform(self.min_dim, self.max_dim)
        select_dim = int(dim_ratio * time_series.shape[-1])
        selected_indices = np.random.choice(time_series.shape[-1], select_dim, replace=False)
        for i in selected_indices:
            time_series[mask, i] = 0
        return time_series

    def window_slicing(self, time_series):
        """Window slicing augmentation.
        # https://halshs.archives-ouvertes.fr/halshs-01357973/document

        Args:
            time_series (torch.Tensor): Window time series.

        Returns:
            torch.Tensor: Augmented time series.
        """
        target_len = np.ceil(self.reduce_ratio*time_series.shape[0])
        if target_len >= time_series.shape[0]: # Return the time series without slicing
            return time_series

        # Else  randomly select a shorter part of the time series
        starts = np.random.randint(low=0, high=time_series.shape[0]-target_len)
        ends = int(target_len + starts)
        #print(starts, ends)
        #ret = np.zeros_like(time_series)

        # Select the dimension to modify
        dim_ratio = random.uniform(self.min_dim, self.max_dim)
        select_dim = int(dim_ratio * time_series.shape[-1])
        selected_indices = np.random.choice(time_series.shape[-1], select_dim, replace=False)

        for dim in selected_indices:
            time_series[:, dim] = torch.tensor(np.interp(np.linspace(0, target_len,
                                                                    num=time_series.shape[0]),
                                    np.arange(target_len),
                                    time_series[starts:ends, dim]
                                    ).T, dtype=torch.float)
        #return torch.tensor(ret, dtype=torch.float)
        return time_series

    def window_warping(self, time_series):
        """Window warping augmentation.
        # https://halshs.archives-ouvertes.fr/halshs-01357973/document

        Args:
            time_series (torch.Tensor): Window time series.

        Returns:
            torch.Tensor: Augmented time series.
        """
        warp_scale = np.random.choice(self.scales)
        warp_size = np.ceil(self.window_ratio*time_series.shape[0])
        window_steps = np.arange(warp_size)

        window_starts = np.random.randint(low=1, high=time_series.shape[0]-warp_size-1)
        window_ends = int(window_starts + warp_size)

        # ret = np.zeros_like(time_series)

        # Select the dimension to modify
        dim_ratio = random.uniform(self.min_dim, self.max_dim)
        select_dim = int(dim_ratio * time_series.shape[-1])
        selected_indices = np.random.choice(time_series.shape[-1], select_dim, replace=False)

        for dim in selected_indices:
            start_seg = time_series[:window_starts,dim]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scale)),
                                    window_steps,
                                    time_series[window_starts:window_ends,dim])
            end_seg = time_series[window_ends:,dim]
            warped = np.concatenate((start_seg, window_seg, end_seg))
            time_series[:,dim] = torch.tensor(np.interp(np.arange(time_series.shape[0]),
                                    np.linspace(0, time_series.shape[0]-1., num=warped.size),
                                    warped
                                    ).T, dtype=torch.float)

        return time_series

    def augment(self, time_series, aug_funct_names=None, positive=True):
        """Apply augmentation to a time series.

        Args:
            time_series (torch.Tensor): Window time series.
            aug_funct_names (list, optional): List of augmentation to apply. Defaults to None.
            positive (bool, optional): Positive augmentation. Defaults to True.

        Raises:
            ValueError: Augmentation function names not implemented.

        Returns:
            torch.Tensor: Augmented time series.
        """
        # Check if is a positive augmentation or not
        if positive:
            augment_dict = self.positives
        else:
            augment_dict = self.negatives

        # Check if the list of augmentation functions exist in accepted augmentation functions.
        if not all(aug_func in augment_dict for aug_func in aug_funct_names):
            raise ValueError(f" Your augmentation functions must be in {list(augment_dict.keys())}")

        # Else apply the list of data augmentation
        aug_func_list = [augment_dict[name] for name in aug_funct_names]
        #print(aug_func_list)

        view_time_series = time_series.clone().detach()
        #print(torch.sum(torch.isnan(view_time_series)).item())
        for aug_func in aug_func_list:
            view_time_series = aug_func(view_time_series)
        #print(torch.sum(torch.isnan(view_time_series)).item())

        return view_time_series

    def random_augment(self, time_series, aug_funct_names=None):
        """Apply random augmentation based on a list of augmentations.

        Args:
            time_series (torch.Tensor): Window time series.
            aug_funct_names (list, optional): List of augmentations to apply.. Defaults to None.

        Raises:
            ValueError: Augmentation function names not implemented.

        Returns:
            torch.Tensor: Augmented time series.
        """
        augment_dict = {**self.positives, **self.negatives}

        # Check if the list of augmentation functions exist in accepted augmentation functions.
        if not all(aug_func in augment_dict for aug_func in aug_funct_names):
            raise ValueError(f" Your augmentation functions must be in {list(augment_dict.keys())}")

        # Randomly choose an augmentation function in the list
        aug_func = random.choice(aug_funct_names)

        view_time_series = augment_dict[aug_func](time_series.clone().detach())
        return view_time_series
