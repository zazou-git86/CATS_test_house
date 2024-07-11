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
from utils.datasets.entities_name import entities_dict


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
                        help='The embedding size. Default is 100.')
    parser.add_argument('--proj-size', type=int, default=50,
                        help='The projection size. Default is 50.')
    parser.add_argument('--crop-ratio-min', type=float, default=.9,
                        help='The minimum crop ratio. Default is 0.5.')
    parser.add_argument('--crop-ratio-max', type=float, default=1,
                        help='The maximum crop ratio. Default is 1.')
    parser.add_argument('--save-dir', type=str,
                        default='outputs/benchmarks',
                        help='The folder to store the model outputs.')
    parser.add_argument('--data-dir', type=str, default='datasets/cg',
                        help='The folder where the data are stored.')

    return parser.parse_args()

def main_entity(args):
    """Benchmark evaluations with entity datasets.

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
    cont_rate = args.contamination_ratio
    out_size = args.out_size
    proj_size = args.proj_size
    crop_ratio_min = args.crop_ratio_min
    crop_ratio_max = args.crop_ratio_max
    #n_runs = args.n_runs
    classic_models = ['iforest', 'ae', 'deep_svdd', 'usad']
    classic_models = []
    #constr_models = ['simclr', 'simsiam', 'ts2vec']
    constr_models = ['simclr', 'simsiam', 'ts2vec', 'cats']
    if data_name not in ['std', 'gfn', 'xc']:
        data_path = os.path.join(data_path, data_name )

    transform = TimeSeriesAugmentation()
    pos_augment_1 = ['jitter']
    pos_augment_2 = ['scaling']
    neg_augment = ['trend', 'spike']
    if data_name not in ['smd', 'smap', 'msl']:
        raise ValueError('This data name is not entity dataset !')
    entity_list = entities_dict[data_name]


    for i in range(len(entity_list)):
        print(f'Dataset {data_name} : {i+1}/{len(entity_list)}=======================================')
        seed = np.random.randint(100)
        dataset = TimeSeriesDataset(data_path=data_path,
                                    win_size=win_size,
                                    stride=stride,
                                    data_name=data_name,
                                    cont_rate=cont_rate,
                                    seed=seed,
                                    entity=True,
                                    entity_number=i,
                                    augmentation=None,
                                    negative=False)

        for model_name in classic_models:
            print(f'Training {model_name} model...')
            outs_dir = os.path.join(save_dir, data_name, model_name, f'model_{i+1}')
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
            outs_dir = os.path.join(save_dir, data_name, model_name, f'model_{i+1}')
            if model_name == 'simclr':
                eval_simclr(dataset, save_dir=outs_dir, output_size=out_size, proj_size=proj_size)
            elif model_name == 'simsiam':
                eval_simsiam(dataset, save_dir=outs_dir, output_size=out_size, proj_size=proj_size)
            elif model_name == 'ts2vec':
                eval_ts2vec(dataset, save_dir=outs_dir, output_size=out_size)
            elif model_name ==  'cats':
                eval_on_data(dataset, save_dir=outs_dir, output_size=out_size, proj_size=proj_size,
                            min_crop_ratio=crop_ratio_min, max_crop_ratio=crop_ratio_max)
            else:
                raise ValueError(f'The model name {model_name} is not implemented !')
            print('Training and evaluation done ! \n')

    # Aggregate and save results 
    models = classic_models + constr_models
    for model_name in models:
        result_path = os.path.join(save_dir, data_name, model_name)
        #_ = aggregate_results_over_runs(result_path, n_runs=len(entity_list), res_type='win')
        _ = aggregate_results_over_runs(result_path, n_runs=len(entity_list), res_type='pw')

def main(args):
    """Benchmark evaluations.

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
    cont_rate = args.contamination_ratio
    n_runs = args.n_runs
    out_size = args.out_size
    proj_size = args.proj_size
    crop_ratio_min = args.crop_ratio_min
    crop_ratio_max = args.crop_ratio_max
    classic_models = ['iforest', 'ae', 'deep_svdd', 'usad']
    #classic_models = []
    #constr_models = ['simclr', 'simsiam', 'ts2vec']
    constr_models = ['simclr', 'simsiam', 'ts2vec', 'cats']
    #constr_models = ['custom']
    if data_name not in ['std', 'gfn', 'xc']:
        data_path = os.path.join(data_path, data_name )

    transform = TimeSeriesAugmentation()
    pos_augment_1 = ['jitter']
    pos_augment_2 = ['scaling']
    neg_augment = ['trend', 'spike']

    print(f'Dataset {data_name}==========================================')

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
            outs_dir = os.path.join(save_dir, data_name, model_name, f'model_{i+1}')
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
            outs_dir = os.path.join(save_dir, data_name, model_name, f'model_{i+1}')
            if model_name == 'simclr':
                eval_simclr(dataset, save_dir=outs_dir, output_size=out_size, proj_size=proj_size)
            elif model_name == 'simsiam':
                eval_simsiam(dataset, save_dir=outs_dir, output_size=out_size, proj_size=proj_size)
            elif model_name == 'ts2vec':
                eval_ts2vec(dataset, save_dir=outs_dir, output_size=out_size)
            elif model_name ==  'cats':
                eval_on_data(dataset, save_dir=outs_dir, output_size=out_size, proj_size=proj_size,
                            min_crop_ratio=crop_ratio_min, max_crop_ratio=crop_ratio_max)
            else:
                raise ValueError(f'The model name {model_name} is not implemented !')
            print('Training and evaluation done ! \n')

    # Aggregate and save results 
    models = classic_models + constr_models
    for model_name in models:
        result_path = os.path.join(save_dir, data_name, model_name)
        #_ = aggregate_results_over_runs(result_path, n_runs=n_runs, res_type='win')
        _ = aggregate_results_over_runs(result_path, n_runs=n_runs, res_type='pw')

if __name__ == '__main__':
    cmd_args = parse_arguments()
    if cmd_args.data_name in ['smd', 'msl', 'smap']:
        main_entity(cmd_args)
    else:
        main(cmd_args)
