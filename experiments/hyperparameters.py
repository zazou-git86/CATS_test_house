"""
Copyright (c) 2024 Orange - All rights reserved

Author:  Joël Roman Ky
This code is distributed under the terms and conditions
of the MIT License (https://opensource.org/licenses/MIT)
"""

import os
import argparse

import numpy as np
import torch

from cats import eval_on_data
from utils.time_series_dataset import TimeSeriesDataset
from utils.augmentation import TimeSeriesAugmentation
from utils.evaluation_utils import aggregate_results_over_runs


torch.manual_seed(0)
all_exps = ['temperature', 'margin', 'gamma', 'batch_size',
                'embed_size', 'proj_size', 'augmentation']

def parse_arguments():
    """Command line parser.

    Returns:
        args: Arguments parsed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--window-size', type=int, default=10,
                        help='The window size. Default is 10.')
    parser.add_argument('--stride', type=int, default=1,
                        help='The stride. Default is 1.')
    parser.add_argument('--contamination-ratio', type=float, default=0.0,
                        help='The contamination ratio. Default is 0.')
    parser.add_argument('--data-name', type=str, default='std',
                        help='The name of the data to use. Default is std.')
    parser.add_argument('--n-runs', type=int, default=5,
                        help='The number of time each experiment is runned. Default is 5.')
    parser.add_argument('--out-size', type=int, default=100,
                        help='The embedding size. Default is 100.')
    parser.add_argument('--proj-size', type=int, default=50,
                        help='The projection size. Default is 50.')
    parser.add_argument('--crop-ratio-min', type=float, default=.5,
                        help='The minimum crop ratio. Default is 0.5.')
    parser.add_argument('--crop-ratio-max', type=float, default=1,
                        help='The maximum crop ratio. Default is 1.')
    parser.add_argument('--experiment', nargs='*', default='temperature', choices=all_exps,
                        help='The hyper parameter experiment to do.')
    parser.add_argument('--save-dir', type=str,
                        default='outputs/hyperparameters',
                        help='The folder to store the model outputs.')
    parser.add_argument('--data-dir', type=str, default='datasets/cg',
                        help='The folder where the data are stored.')

    return parser.parse_args()

def temp_exp(args):
    """Temperature sensitivty experiments.

    Args:
        args (_type_): Arguments from command line.
    """
    data_path = args.data_dir
    save_dir = args.save_dir
    win_size = args.window_size
    stride = args.stride
    data_name = args.data_name
    cont_rate = args.contamination_ratio
    n_runs = args.n_runs
    out_size = args.out_size
    proj_size = args.proj_size
    crop_ratio_min = args.crop_ratio_min
    crop_ratio_max = args.crop_ratio_max

    transform = TimeSeriesAugmentation()
    pos_augment_1 = ['jitter']
    pos_augment_2 = ['scaling']
    neg_augment_list = ['trend']

    temp_list = [.01, .1, .5, 1]
    save_dir = os.path.join(save_dir, 'temp')


    for temp in temp_list:
        print(f"Temperature = {temp}")
        for i in range(n_runs):
            seed = np.random.randint(100)
            dataset = TimeSeriesDataset(data_path=data_path,
                                        win_size=win_size,
                                        stride=stride,
                                        data_name=data_name,
                                        cont_rate=cont_rate,
                                        seed=seed,
                                        augmentation=transform,
                                        pos_augment_1=pos_augment_1,
                                        pos_augment_2=pos_augment_2,
                                        neg_augment_func_list=neg_augment_list,
                                        negative=True)

            outs_dir = os.path.join(save_dir, data_name, f'{temp}', f'model_{i+1}')
            eval_on_data(dataset, save_dir=outs_dir, output_size=out_size,
                        proj_size=proj_size, min_crop_ratio=crop_ratio_min,
                        max_crop_ratio=crop_ratio_max, temperature=temp)

            print('Training and evaluation done ! \n')

        # Aggregate and save results
        #for augment_remove in neg_augment_list:
        result_path = os.path.join(save_dir, data_name, f'{temp}')
        #_ = aggregate_results_over_runs(result_path, n_runs=n_runs, res_type='win')
        _ = aggregate_results_over_runs(result_path, n_runs=n_runs, res_type='pw')

def gamma_exp(args):
    """Gamma sensitivty experiments.

    Args:
        args (_type_): Arguments from command line.
    """
    data_path = args.data_dir
    save_dir = args.save_dir
    win_size = args.window_size
    stride = args.stride
    data_name = args.data_name
    cont_rate = args.contamination_ratio
    n_runs = args.n_runs
    out_size = args.out_size
    proj_size = args.proj_size
    crop_ratio_min = args.crop_ratio_min
    crop_ratio_max = args.crop_ratio_max

    transform = TimeSeriesAugmentation()
    pos_augment_1 = ['jitter']
    pos_augment_2 = ['scaling']
    neg_augment_list = ['trend']

    gamma_list = [.01, .1, .5, 1]
    save_dir = os.path.join(save_dir, 'gamma')


    for gamma in gamma_list:
        print(f"Gamma = {gamma}")
        for i in range(n_runs):
            seed = np.random.randint(100)
            dataset = TimeSeriesDataset(data_path=data_path,
                                        win_size=win_size,
                                        stride=stride,
                                        data_name=data_name,
                                        cont_rate=cont_rate,
                                        seed=seed,
                                        augmentation=transform,
                                        pos_augment_1=pos_augment_1,
                                        pos_augment_2=pos_augment_2,
                                        neg_augment_func_list=neg_augment_list,
                                        negative=True)

            outs_dir = os.path.join(save_dir, data_name, f'{gamma}', f'model_{i+1}')
            eval_on_data(dataset, save_dir=outs_dir, output_size=out_size,
                        proj_size=proj_size, min_crop_ratio=crop_ratio_min,
                        max_crop_ratio=crop_ratio_max, gamma=gamma)

            print('Training and evaluation done ! \n')

        # Aggregate and save results
        #for augment_remove in neg_augment_list:
        result_path = os.path.join(save_dir, data_name, f'{gamma}')
        #_ = aggregate_results_over_runs(result_path, n_runs=n_runs, res_type='win')
        _ = aggregate_results_over_runs(result_path, n_runs=n_runs, res_type='pw')

def proj_size_exp(args):
    """Projection size sensitivty experiments.

    Args:
        args (_type_): Arguments from command line.
    """
    data_path = args.data_dir
    save_dir = args.save_dir
    win_size = args.window_size
    stride = args.stride
    data_name = args.data_name
    cont_rate = args.contamination_ratio
    n_runs = args.n_runs
    out_size = args.out_size
    #proj_size = args.proj_size
    crop_ratio_min = args.crop_ratio_min
    crop_ratio_max = args.crop_ratio_max

    transform = TimeSeriesAugmentation()
    pos_augment_1 = ['jitter']
    pos_augment_2 = ['scaling']
    neg_augment_list = ['trend']

    proj_size_list = [16, 32, 64, 128, 256]
    save_dir = os.path.join(save_dir, 'proj_size')


    for proj_size in proj_size_list:
        print(f"Projection size = {proj_size}")
        for i in range(n_runs):
            seed = np.random.randint(100)
            dataset = TimeSeriesDataset(data_path=data_path,
                                        win_size=win_size,
                                        stride=stride,
                                        data_name=data_name,
                                        cont_rate=cont_rate,
                                        seed=seed,
                                        augmentation=transform,
                                        pos_augment_1=pos_augment_1,
                                        pos_augment_2=pos_augment_2,
                                        neg_augment_func_list=neg_augment_list,
                                        negative=True)

            outs_dir = os.path.join(save_dir, data_name, f'{proj_size}', f'model_{i+1}')
            eval_on_data(dataset, save_dir=outs_dir, output_size=out_size,
                        proj_size=proj_size, min_crop_ratio=crop_ratio_min,
                        max_crop_ratio=crop_ratio_max)

            print('Training and evaluation done ! \n')

        # Aggregate and save results
        #for augment_remove in neg_augment_list:
        result_path = os.path.join(save_dir, data_name, f'{proj_size}')
        #_ = aggregate_results_over_runs(result_path, n_runs=n_runs, res_type='win')
        _ = aggregate_results_over_runs(result_path, n_runs=n_runs, res_type='pw')

def output_size_exp(args):
    """Output size sensitivty experiments.

    Args:
        args (_type_): Arguments from command line.
    """
    data_path = args.data_dir
    save_dir = args.save_dir
    win_size = args.window_size
    stride = args.stride
    data_name = args.data_name
    cont_rate = args.contamination_ratio
    n_runs = args.n_runs
    #out_size = args.out_size
    proj_size = args.proj_size
    crop_ratio_min = args.crop_ratio_min
    crop_ratio_max = args.crop_ratio_max

    transform = TimeSeriesAugmentation()
    pos_augment_1 = ['jitter']
    pos_augment_2 = ['scaling']
    neg_augment_list = ['trend']

    output_size_list = [16, 32, 64, 128, 256]
    save_dir = os.path.join(save_dir, 'output_size')


    for out_size in output_size_list:
        print(f"Output size = {out_size}")
        for i in range(n_runs):
            seed = np.random.randint(100)
            dataset = TimeSeriesDataset(data_path=data_path,
                                        win_size=win_size,
                                        stride=stride,
                                        data_name=data_name,
                                        cont_rate=cont_rate,
                                        seed=seed,
                                        augmentation=transform,
                                        pos_augment_1=pos_augment_1,
                                        pos_augment_2=pos_augment_2,
                                        neg_augment_func_list=neg_augment_list,
                                        negative=True)

            outs_dir = os.path.join(save_dir, data_name, f'{out_size}', f'model_{i+1}')
            eval_on_data(dataset, save_dir=outs_dir, output_size=out_size,
                        proj_size=proj_size, min_crop_ratio=crop_ratio_min,
                        max_crop_ratio=crop_ratio_max)

            print('Training and evaluation done ! \n')

        # Aggregate and save results
        #for augment_remove in neg_augment_list:
        result_path = os.path.join(save_dir, data_name, f'{out_size}')
        #_ = aggregate_results_over_runs(result_path, n_runs=n_runs, res_type='win')
        _ = aggregate_results_over_runs(result_path, n_runs=n_runs, res_type='pw')

def batch_size_exp(args):
    """Batch size sensitivty experiments.

    Args:
        args (_type_): Arguments from command line.
    """
    data_path = args.data_dir
    save_dir = args.save_dir
    win_size = args.window_size
    stride = args.stride
    data_name = args.data_name
    cont_rate = args.contamination_ratio
    n_runs = args.n_runs
    out_size = args.out_size
    proj_size = args.proj_size
    crop_ratio_min = args.crop_ratio_min
    crop_ratio_max = args.crop_ratio_max

    transform = TimeSeriesAugmentation()
    pos_augment_1 = ['jitter']
    pos_augment_2 = ['scaling']
    neg_augment_list = ['trend']

    batch_size_list = [64, 128, 256, 512]
    save_dir = os.path.join(save_dir, 'batch_size')


    for batch in batch_size_list:
        print(f"Batch size = {batch}")
        for i in range(n_runs):
            seed = np.random.randint(100)
            dataset = TimeSeriesDataset(data_path=data_path,
                                        win_size=win_size,
                                        stride=stride,
                                        data_name=data_name,
                                        cont_rate=cont_rate,
                                        seed=seed,
                                        augmentation=transform,
                                        pos_augment_1=pos_augment_1,
                                        pos_augment_2=pos_augment_2,
                                        neg_augment_func_list=neg_augment_list,
                                        negative=True)

            outs_dir = os.path.join(save_dir, data_name, f'{batch}', f'model_{i+1}')
            eval_on_data(dataset, save_dir=outs_dir, output_size=out_size, batch_size=batch,
                        proj_size=proj_size, min_crop_ratio=crop_ratio_min,
                        max_crop_ratio=crop_ratio_max)

            print('Training and evaluation done ! \n')

        # Aggregate and save results
        #for augment_remove in neg_augment_list:
        result_path = os.path.join(save_dir, data_name, f'{batch}')
        #_ = aggregate_results_over_runs(result_path, n_runs=n_runs, res_type='win')
        _ = aggregate_results_over_runs(result_path, n_runs=n_runs, res_type='pw')

def margin_exp(args):
    """Margin sensitivty experiments.

    Args:
        args (_type_): Arguments from command line.
    """
    data_path = args.data_dir
    save_dir = args.save_dir
    win_size = args.window_size
    stride = args.stride
    data_name = args.data_name
    cont_rate = args.contamination_ratio
    n_runs = args.n_runs
    out_size = args.out_size
    proj_size = args.proj_size
    crop_ratio_min = args.crop_ratio_min
    crop_ratio_max = args.crop_ratio_max

    transform = TimeSeriesAugmentation()
    pos_augment_1 = ['jitter']
    pos_augment_2 = ['scaling']
    neg_augment_list = ['trend']

    margin_list = [1, 2, 5, 10]
    save_dir = os.path.join(save_dir, 'margin')


    for margin in margin_list:
        print(f"Margin size = {margin}")
        for i in range(n_runs):
            seed = np.random.randint(100)
            dataset = TimeSeriesDataset(data_path=data_path,
                                        win_size=win_size,
                                        stride=stride,
                                        data_name=data_name,
                                        cont_rate=cont_rate,
                                        seed=seed,
                                        augmentation=transform,
                                        pos_augment_1=pos_augment_1,
                                        pos_augment_2=pos_augment_2,
                                        neg_augment_func_list=neg_augment_list,
                                        negative=True)

            outs_dir = os.path.join(save_dir, data_name, f'{margin}', f'model_{i+1}')
            eval_on_data(dataset, save_dir=outs_dir, output_size=out_size, margin=margin,
                        proj_size=proj_size, min_crop_ratio=crop_ratio_min,
                        max_crop_ratio=crop_ratio_max)

            print('Training and evaluation done ! \n')

        # Aggregate and save results
        #for augment_remove in neg_augment_list:
        result_path = os.path.join(save_dir, data_name, f'{margin}')
        #_ = aggregate_results_over_runs(result_path, n_runs=n_runs, res_type='win')
        _ = aggregate_results_over_runs(result_path, n_runs=n_runs, res_type='pw')

def augmentation_exp(args):
    """Data augmentation experiments.

    Args:
        args (_type_): Arguments from command line.
    """
    data_path = args.data_dir
    save_dir = args.save_dir
    win_size = args.window_size
    stride = args.stride
    data_name = args.data_name
    cont_rate = args.contamination_ratio
    n_runs = args.n_runs
    out_size = args.out_size
    proj_size = args.proj_size
    crop_ratio_min = args.crop_ratio_min
    crop_ratio_max = args.crop_ratio_max

    transform = TimeSeriesAugmentation()
    pos_augment_1 = ['jitter']
    pos_augment_2 = ['scaling']
    neg_augment_list = ['trend', 'spike', 'masking', 'shuffle']
    save_dir = os.path.join(save_dir, 'augmentations')
    n = len(neg_augment_list)
    couple_aug_neg = []
    for i in range(n):
        couple_aug_neg.append([neg_augment_list[i]])
        for j in range(i+1, n):
            couple = [neg_augment_list[i], neg_augment_list[j]]
            couple_aug_neg.append(couple)

    #entity_number = 1
    for _, couple in enumerate(couple_aug_neg):
        if len(couple)==2:
            augment_apply = f'{couple[0]}_{couple[1]}'
        elif len(couple)==1:
            augment_apply = f'{couple[0]}'
        else:
            raise ValueError('Unexpected negative data augmentation encountered')
        print(f'Augmentation with {augment_apply}')
        for i in range(n_runs):
            seed = np.random.randint(100)
            dataset = TimeSeriesDataset(data_path=data_path,
                                        win_size=win_size,
                                        stride=stride,
                                        data_name=data_name,
                                        cont_rate=cont_rate,
                                        seed=seed,
                                        augmentation=transform,
                                        pos_augment_1=pos_augment_1,
                                        pos_augment_2=pos_augment_2,
                                        neg_augment_func_list=couple,
                                        negative=True)

            outs_dir = os.path.join(save_dir, data_name, augment_apply, f'model_{i+1}')
            eval_on_data(dataset, save_dir=outs_dir, output_size=out_size,
                        proj_size=proj_size, min_crop_ratio=crop_ratio_min,
                        max_crop_ratio=crop_ratio_max)

            print('Training and evaluation done ! \n')

        # Aggregate and save results
        #for augment_remove in neg_augment_list:
        result_path = os.path.join(save_dir, data_name, augment_apply)
        #_ = aggregate_results_over_runs(result_path, n_runs=n_runs, res_type='win')
        _ = aggregate_results_over_runs(result_path, n_runs=n_runs, res_type='pw')

if __name__ == '__main__':
    cmd_args = parse_arguments()
    experiment = cmd_args.experiment
    if experiment=='batch_size':
        batch_size_exp(cmd_args)
    elif experiment=='embed_size':
        output_size_exp(cmd_args)
    elif experiment=='proj_size':
        proj_size_exp(cmd_args)
    elif experiment=='temperature':
        temp_exp(cmd_args)
    elif experiment=='gamma':
        gamma_exp(cmd_args)
    elif experiment=='margin':
        margin_exp(cmd_args)
    elif experiment=='augmentation':
        augmentation_exp(cmd_args)
    else:
        raise ValueError(f'Experiment name must be in {all_exps}')
