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

from models.simclr import eval_on_data as eval_simclr
from cats import eval_on_data
from utils.time_series_dataset import TimeSeriesDataset
from utils.augmentation import TimeSeriesAugmentation
from utils.evaluation_utils import aggregate_results_over_runs


torch.manual_seed(0)

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
                        help='The embedding size. Default is 128.')
    parser.add_argument('--proj-size', type=int, default=50,
                        help='The projection size. Default is 128.')
    parser.add_argument('--crop-ratio-min', type=float, default=.5,
                        help='The minimum crop ratio. Default is 0.5.')
    parser.add_argument('--crop-ratio-max', type=float, default=1,
                        help='The maximum crop ratio. Default is 1.')
    parser.add_argument('--save-dir', type=str,
                        default='outputs/ablations',
                        help='The folder to store the model outputs.')
    parser.add_argument('--data-dir', type=str, default='datasets/cg',
                        help='The folder where the data are stored.')

    return parser.parse_args()

def main(args):
    """Ablation study.

    Args:
        args (_type_): Arguments from command line.

    Raises:
        ValueError: Ablation type not implemented.
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
    neg_augment = ['trend', 'spike']

    ablations_type = ['ntxent', 'tcl_crop', 'gcl', 'all', 'tcl_gcl_wo_crop']

    #entity_number = 1
    print(f'Ablation study on dataset : {data_name}\n')
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
                                    neg_augment_func_list=neg_augment,
                                    negative=True)

        # dataset = TimeSeriesDataset(data_path=data_path,
        #                 win_size=win_size,
        #                 stride=stride,
        #                 data_name=data_name,
        #                 cont_rate=cont_rate,
        #                 seed=seed,
        #                 entity=True,
        #                 entity_number=entity_number,
        #                 augmentation=transform,
        #                 pos_augment_1=pos_augment_1,
        #                 pos_augment_2=pos_augment_2,
        #                 neg_augment_func_list=neg_augment,
        #                 negative=True)

        for ablation in ablations_type:
            print(f'Training with ablations: {ablation} ...\n')
            outs_dir = os.path.join(save_dir, data_name, ablation, f'model_{i+1}')
            if ablation == 'tcl_wo_crop':
                eval_on_data(dataset, save_dir=outs_dir, output_size=out_size, proj_size=proj_size,
                            min_crop_ratio=crop_ratio_min, max_crop_ratio=crop_ratio_max,
                            tcl_inst=True, gcl_inst=False, crop=False,)
            elif ablation =='ntxent':
                eval_simclr(dataset, save_dir=outs_dir, output_size=out_size, proj_size=proj_size)
            elif ablation == 'tcl_crop':
                eval_on_data(dataset, save_dir=outs_dir, output_size=out_size, proj_size=proj_size,
                            min_crop_ratio=crop_ratio_min, max_crop_ratio=crop_ratio_max,
                            tcl_inst=True, gcl_inst=False, crop=True,)
            elif ablation == 'gcl':
                eval_on_data(dataset, save_dir=outs_dir, output_size=out_size, proj_size=proj_size,
                            min_crop_ratio=crop_ratio_min, max_crop_ratio=crop_ratio_max,
                            tcl_inst=False, gcl_inst=True, crop=False,)
            elif ablation == 'tcl_gcl_wo_crop':
                eval_on_data(dataset, save_dir=outs_dir, output_size=out_size, proj_size=proj_size,
                            min_crop_ratio=crop_ratio_min, max_crop_ratio=crop_ratio_max,
                            tcl_inst=True, gcl_inst=True, crop=False,)
            elif ablation == 'all':
                eval_on_data(dataset, save_dir=outs_dir, output_size=out_size, proj_size=proj_size,
                            min_crop_ratio=crop_ratio_min, max_crop_ratio=crop_ratio_max,
                            tcl_inst=True, gcl_inst=True, crop=True,)
            else:
                raise ValueError("This ablation type is not implemented")
            print('Training and evaluation done ! \n')

    # Aggregate and save results
    for ablation in ablations_type:
        result_path = os.path.join(save_dir, data_name, ablation)
        #_ = aggregate_results_over_runs(result_path, n_runs=n_runs, res_type='win')
        _ = aggregate_results_over_runs(result_path, n_runs=n_runs, res_type='pw')

if __name__ == '__main__':
    cmd_args = parse_arguments()
    main(cmd_args)
