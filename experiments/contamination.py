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

from models.iforest import eval_on_data as eval_forest
from models.auto_encoder import eval_on_data as eval_ae
from models.deep_svdd import eval_on_data as eval_svdd
from models.usad import eval_on_data as eval_usad
from models.simclr import eval_on_data as eval_simclr
from models.simsiam import eval_on_data as eval_simsiam
from models.ts2vec import eval_on_data as eval_ts2vec
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
                        default='outputs/contamination',
                        help='The folder to store the model outputs.')
    parser.add_argument('--data-dir', type=str, default='datasets/cg',
                        help='The folder where the data are stored.')

    return parser.parse_args()

def main(args):
    """Data contamination evaluations.

    Args:
        args (_type_): Arguments from command line.

    Raises:
        ValueError: Model name not implemented.
    """
    data_path = args.data_dir
    save_dir = args.save_dir
    win_size = args.window_size
    stride = args.stride
    data_name = args.data_name
    n_runs = args.n_runs
    out_size = args.out_size
    proj_size = args.proj_size
    crop_ratio_min = args.crop_ratio_min
    crop_ratio_max = args.crop_ratio_max
    classic_models = ['iforest', 'ae', 'deep_svdd', 'usad']
    classic_models = []
    #constr_models = ['simclr', 'simsiam', 'ts2vec']
    constr_models = ['simclr', 'simsiam', 'ts2vec','cats']

    transform = TimeSeriesAugmentation()
    pos_augment_1 = ['jitter']
    pos_augment_2 = ['scaling']
    neg_augment = ['trend']

    contamination_ratio = [.04, .08, .12, .2]

    for cont_rate in contamination_ratio:
        print(f'Contamination rate of {cont_rate*100} %')
        for i in range(n_runs):
            seed = np.random.randint(100)
            dataset = TimeSeriesDataset(data_path=data_path,
                                        win_size=win_size,
                                        stride=stride,
                                        data_name=data_name,
                                        cont_rate=cont_rate,
                                        seed=seed,
                                        augmentation=None,
                                        negative=False)

            for model_name in classic_models:
                print(f'Training {model_name} model...')
                outs_dir = os.path.join(save_dir, data_name, model_name,
                                        f'cont_{cont_rate*100}', f'model_{i+1}')
                if model_name == 'iforest':
                    eval_forest(dataset, save_dir=outs_dir)
                elif model_name == 'ae':
                    eval_ae(dataset, save_dir=outs_dir)
                elif model_name == 'deep_svdd':
                    eval_svdd(dataset, save_dir=outs_dir)
                elif model_name == 'usad':
                    eval_usad(dataset, save_dir=outs_dir)
                else:
                    raise ValueError(f'The model name {model_name} is not implemented !')
                print('Training and evaluation done ! \n')

            # Set augmentation for contrastive models
            dataset.set_augmentation(transform, pos_augment_1=pos_augment_1,
                                    pos_augment_2=pos_augment_2, neg_augment_func_list=neg_augment,
                                    negative=True)

            for model_name in constr_models:
                print(f'Training {model_name} model...')
                outs_dir = os.path.join(save_dir, data_name, model_name,
                                        f'cont_{cont_rate*100}', f'model_{i+1}')
                if model_name == 'simclr':
                    eval_simclr(dataset, save_dir=outs_dir, output_size=out_size,
                                proj_size=proj_size)
                elif model_name == 'simsiam':
                    eval_simsiam(dataset, save_dir=outs_dir, output_size=out_size,
                                proj_size=proj_size)
                elif model_name == 'ts2vec':
                    eval_ts2vec(dataset, save_dir=outs_dir, output_size=out_size)
                elif model_name ==  'cats':
                    eval_on_data(dataset, save_dir=outs_dir, output_size=out_size,
                                proj_size=proj_size, min_crop_ratio=crop_ratio_min,
                                max_crop_ratio=crop_ratio_max)
                # elif model_name ==  'custom_gcl':
                #     eval_on_data(dataset, save_dir=outs_dir, output_size=out_size,
                #                 proj_size=proj_size, min_crop_ratio=crop_ratio_min,
                #                 max_crop_ratio=crop_ratio_max, tcl_inst=False)
                # elif model_name ==  'custom_tcl':
                #     eval_on_data(dataset, save_dir=outs_dir, output_size=out_size,
                #                 proj_size=proj_size, min_crop_ratio=.5,
                #                 max_crop_ratio=1, gcl_inst=False)
                else:
                    raise ValueError(f'The model name {model_name} is not implemented !')
                print('Training and evaluation done ! \n')

    # Aggregate and save results
    models = classic_models + constr_models
    for model_name in models:
        for cont_rate in contamination_ratio:
            result_path = os.path.join(save_dir, data_name, model_name, f'cont_{cont_rate*100}')
            #_ = aggregate_results_over_runs(result_path, n_runs=n_runs, res_type='win')
            _ = aggregate_results_over_runs(result_path, n_runs=n_runs, res_type='pw')

if __name__ == '__main__':
    cmd_args = parse_arguments()
    main(cmd_args)
