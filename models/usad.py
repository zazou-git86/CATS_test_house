"""
Copyright (c) 2024 Orange - All rights reserved

Author:  Joël Roman Ky
This code is distributed under the terms and conditions
of the MIT License (https://opensource.org/licenses/MIT)
"""

import logging
import sys
from typing import Tuple, List
import pickle
import os

from tqdm import trange
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from utils.algorithm_utils import AverageMeter, get_train_val_data_loaders
from utils.evaluation_utils import compute_results_csv



# Get logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class USADModel(nn.Module):
    """USAD neural network architecture.
    """
    def __init__(self, input_length: int,
                hidden_size: int):
        """USAD model architecture.

        Args:
            input_length (int)      : The number of input features.
            hidden_size (int)       : The hidden size.
            seed (int)              : The random generator seed.
            gpu (int)               : The number of the GPU device.
        """
        # Each point is a flattened window and thus has as many features
        # as sequence_length * features
        super().__init__()
        # input_length = n_features * sequence_length

        enc_layers = [
            (input_length, input_length//2, nn.ReLU()),
            (input_length//2, input_length//4, nn.ReLU()),
            (input_length//4, hidden_size, nn.ReLU())
        ]
        dec_layers = [
            (hidden_size, input_length//4, nn.ReLU()),
            (input_length//4, input_length//2, nn.ReLU()),
            (input_length//2, input_length, nn.Sigmoid())
        ]

        self.encoder = self._make_linear(enc_layers)
        self.decoder_1 = self._make_linear(dec_layers)
        self.decoder_2 = self._make_linear(dec_layers)

    def _make_linear(self, layers: List[Tuple]):
        """
        This function builds a linear model whose units and layers depend on
        the passed @layers argument
        :param layers: a list of tuples indicating the layers architecture
        (in_neuron, out_neuron, activation_function)
        :return: a fully connected neural net (Sequentiel object)
        """
        net_layers = []
        for in_neuron, out_neuron, act_fn in layers:
            net_layers.append(nn.Linear(in_neuron, out_neuron))
            if act_fn:
                net_layers.append(act_fn)
        return nn.Sequential(*net_layers)

    def forward(self, ts_batch: torch.Tensor):
        """Forward function.

        Args:
            ts_batch (torch.Tensor): Batch tensor.

        Returns:
            torch.Tensor: Model output.
        """
        # if ts_batch.dim() == 3:
        #     b_size = ts_batch.shape[0]
        #     ts_batch = ts_batch.reshape(b_size, -1)
        z_tensor = self.encoder(ts_batch)
        outs1 = self.decoder_1(z_tensor)
        outs2 = self.decoder_2(z_tensor)
        outs2_t = self.decoder_2(self.encoder(outs1))
        return outs1, outs2, outs2_t

    def compute_loss(self, ts_batch, outs1, outs2, outs2_t, train_epoch):
        """Compute loss function

        Args:
            ts_batch (torch.Tensor): Batch inputs.
            outs1 (torch.Tensor): Outputs from AE 1.
            outs2 (torch.Tensor): Outputs from AE 2.
            outs2_t (torch.Tensor): Outputs from boths AEs.
            train_epoch (int): Epoch number.

        Returns:
            torch.Tensor: Training loss.
        """
        loss_fn = torch.nn.MSELoss()
        recons_err_1 = loss_fn(outs1, ts_batch)
        recons_err_2 = loss_fn(outs2, ts_batch)
        advers_err = loss_fn(outs2_t, ts_batch)
        loss_1 = 1/train_epoch * recons_err_1 + (1 - 1/train_epoch)*advers_err
        loss_2 = 1/train_epoch * recons_err_2 - (1 - 1/train_epoch)*advers_err
        return loss_1, loss_2

    def compute_test_score(self, ts_batch, outs1, outs2):
        """Compute test score.

        Args:
            ts_batch (torch.Tensor): Batch inputs.
            outs1 (torch.Tensor): Outputs from AE 1.
            outs2 (torch.Tensor): Outputs from AE 2.

        Returns:
            torch.Tensor: Anomaly score.
        """
        # loss_fn = torch.nn.MSELoss()
        recons_err_1 = torch.mean((ts_batch - outs1)**2, axis=-1) #loss_fn(w1, ts_batch)
        recons_err_2 = torch.mean((ts_batch - outs2)**2, axis=-1) #loss_fn(w2, ts_batch)
        # recons_err_1 = (ts_batch - outs1)**2 #loss_fn(w1, ts_batch)
        # recons_err_2 = (ts_batch - outs2)**2 #loss_fn(w2, ts_batch)

        return recons_err_1, recons_err_2

def fit_with_early_stopping(train_loader, val_loader, model, device,
                            patience, num_epochs,
                            learning_rate, writer, verbose=True):
    """The fitting function of the Auto Encoder.

    Args:
        train_loader (Dataloader)   : The train dataloader.
        val_loader (Dataloader)     : The val dataloader.
        model (nn.Module)           : The Pytorch model.
        device (torch.device)       : The model device.                  
        patience (int)              : The number of epochs to wait for early stopping.
        num_epochs (int)            : The max number of epochs.
        lr (float)                  : The learning rate.
        writer (SummaryWriter)      : The Tensorboard Summary Writer.
        verbose (bool, optional)    : Defaults to True.

    Returns:
                        [nn.Module ]: The fitted model.
    """
    model.to(device)  # .double()
    optimizer_1 = torch.optim.Adam(list(model.encoder.parameters()) + \
                                list(model.decoder_1.parameters()), lr=learning_rate)
    optimizer_2 = torch.optim.Adam(list(model.encoder.parameters()) + \
                                list(model.decoder_2.parameters()), lr=learning_rate)

    model.train()
    #train_loss_by_epoch = []
    #val_loss_by_epoch = []
    best_val_loss = np.inf
    epoch_wo_improv = 0
    best_params = model.state_dict()
    # assuming first batch is complete
    for epoch in trange(num_epochs):
        # If improvement continue training
        if epoch_wo_improv < patience:
            # logging.debug(f'Epoch {epoch + 1}/{num_epochs}.')
            logging.debug('Epoch %d/%d.', epoch + 1, num_epochs)
            #if verbose:
                #GPUtil.showUtilization()
            # Train the model
            #logger.debug("Begin training...")
            train_loss_1, train_loss_2 = train(train_loader,
                                                model,
                                                optimizer_1,
                                                optimizer_2,
                                                epoch, device)

            # Get Validation loss
            #logger.debug("Begin evaluation")
            val_loss_1, val_loss_2 = validation(val_loader, model, epoch, device)

            if verbose:
                # logger.info(f"Epoch: [{epoch+1}/{num_epochs}] - \
                #             Train loss 1: {train_loss_1:.2f} /
                #             Train loss 2: {train_loss_2:.2f} \
                #             - Val loss 1: {val_loss_1:.2f} / Val loss 2: {val_loss_2:.2f}")
                logger.info("Epoch: [%d/%d] - Train loss 1: %2f / Train loss 2: %2f - \
                            Val loss 1: %2f / Val loss 2: %2f",
                            epoch+1, num_epochs, train_loss_1, train_loss_2,
                            val_loss_1, val_loss_2)

            # Write in TensorBoard
            writer.add_scalar('train_loss_1', train_loss_1, epoch)
            writer.add_scalar('val_loss_1', val_loss_1, epoch)
            writer.add_scalar('train_loss_2', train_loss_2, epoch)
            writer.add_scalar('val_loss_2', val_loss_2, epoch)

            # Check if the validation loss improve or not
            if val_loss_1 + val_loss_2 < best_val_loss:
                best_val_loss = val_loss_1 + val_loss_2
                epoch_wo_improv = 0
                best_params = model.state_dict()
            elif val_loss_1 + val_loss_2 >= best_val_loss:
                epoch_wo_improv += 1

        else:
            # No improvement => early stopping is applied and best model is kept
            model.load_state_dict(best_params)
            break

    return model


def train(train_loader, model, optimizer_1, optimizer_2, epoch, device):
    """The training step.

    Args:
        train_loader (Dataloader)       : The train data loader.
        model (nn.Module)               : The Pytorch model.
        optimizer_1 (torch.optim)       : The Optimizer of first AE.
        optimizer_2 (torch.optim)       : The Optimizer of second AE.
        epoch (int)                     : The max number of epochs.
        device (torch.device)           : The model device.  

    Returns:
                The average loss on the epoch.
    """
    # Compute statistics
    loss1_meter = AverageMeter()
    loss2_meter = AverageMeter()

    model.train()
    for ts_batch in train_loader:
        ts_batch = ts_batch.to(device)
        # if ts_batch.dim() == 3:
        #     b_size = ts_batch.shape[0]
        #     ts_batch = ts_batch.reshape(b_size, -1)
            # f_size = ts_batch.shape[-1]
            # ts_batch = ts_batch.reshape(-1, f_size)


        outs1, outs2, outs2_t = model(ts_batch)
        loss1, loss2 = model.compute_loss(ts_batch, outs1, outs2, outs2_t, epoch+1)

        loss = loss1 + loss2

        loss.backward()

        optimizer_1.step()
        optimizer_1.zero_grad()
        optimizer_2.step()
        optimizer_2.zero_grad()

        # multiplying by length of batch to correct accounting for incomplete batches
        loss1_meter.update(loss1.item())
        loss2_meter.update(loss2.item())
        #train_loss.append(loss.item()*len(ts_batch))

    #train_loss = np.mean(train_loss)/train_loader.batch_size
    #train_loss_by_epoch.append(loss_meter.avg)

    return loss1_meter.avg, loss2_meter.avg

def validation(val_loader, model, epoch, device):
    """The validation step.

    Args:
        val_loader (Dataloader)         : The val data loader.
        model (nn.Module)               : The Pytorch model.
        optimizer (torch.optim)         : The Optimizer.
        epoch (int)                     : The max number of epochs.
        device (torch.device)           : The model device.  

    Returns:
                The average loss on the epoch.
    """

    # Compute statistics
    loss1_meter = AverageMeter()
    loss2_meter = AverageMeter()

    model.eval()
    #val_loss = []
    with torch.no_grad():
        for ts_batch in val_loader:
            ts_batch = ts_batch.to(device)
            # if ts_batch.dim() == 3:
            #     b_size = ts_batch.shape[0]
            #     ts_batch = ts_batch.reshape(b_size, -1)
                # f_size = ts_batch.shape[-1]
                # ts_batch = ts_batch.reshape(-1, f_size)

            outs1, outs2, outs2_t = model(ts_batch)
            loss1, loss2 = model.compute_loss(ts_batch, outs1, outs2, outs2_t, epoch+1)

            # multiplying by length of batch to correct accounting for incomplete batches
            loss1_meter.update(loss1.item())
            loss2_meter.update(loss2.item())
        return loss1_meter.avg, loss2_meter.avg

@torch.no_grad()
def predict_test_scores(model, test_loader, device, alpha=.5, beta=.5):
    """The prediction step.

    Args:
        model (nn.Module)               : The PyTorch model.
        test_loader (Dataloader)        : The test dataloader.
        device (torch.device)           : The model device.  
        alpha (float, optional)         : USAD trade-off parameters. Defaults to 0.5.
        beta (float, optional)          : USAD trade-off parameters. Defaults to 0.5.

    Returns:
                The reconstruction score 
    """
    model.eval()
    reconstr_scores = []
    labels_list = []

    for data in test_loader:
        ts_batch, label = data
        ts_batch, label = ts_batch.to(device), label.to(device)
        # if ts_batch.dim() == 3:
        #     b_size = ts_batch.shape[0]
        #     ts_batch = ts_batch.reshape(b_size, -1)
            # f_size = ts_batch.shape[-1]
            # ts_batch = ts_batch.reshape(-1, f_size)
        # ts_batch = ts_batch.float().to(model.device)

        outs1 = model.decoder_1(model.encoder(ts_batch))
        outs2 = model.decoder_2(model.encoder(outs1))

        score1, score2 = model.compute_test_score(ts_batch, outs1, outs2)
        #print(score1.shape)
        score = alpha*score1 + beta*score2
        # print(score.shape)
        # score = score.reshape(-1, 10)
        # print(score.shape)
        # score = torch.mean(score, dim=-1)
        reconstr_scores.append(score.cpu().numpy().tolist())

        labels_list.append(label.cpu().numpy())
    reconstr_scores = np.concatenate(reconstr_scores)
    labels_list = np.concatenate(labels_list, axis=0)
    return reconstr_scores, labels_list


def eval_on_data(dataset, save_dir, batch_size=512, learning_rate=.001,
                num_epochs=100, patience=10, hidden_size=40,
                use_gpu=True):
    """Train and evaluate the model on a dataset.

    Args:
        dataset (torch.Dataset): Dataset.
        save_dir (str): Path to the save directory.
        batch_size (int, optional): Batch size. Defaults to 512.
        learning_rate (float, optional): Learning rate. Defaults to .001.
        num_epochs (int, optional): Number of epochs. Defaults to 100.
        patience (int, optional): Number of epochs to wait before stopping. Defaults to 10.
        hidden_size (int, optional): Embedding size. Defaults to 40.
        use_gpu (bool, optional): Use GPU. Defaults to True.
    """

    win_size = dataset.win_size #data_dict['train']['x'].shape[1]
    feat_size = dataset.n_features #data_dict['train']['x'].shape[2]
    input_length = feat_size#*win_size

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    out_path = os.path.join(save_dir, 'models')
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    if use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    writer = SummaryWriter(log_dir=out_path)


    dataset.set_flag('train')
    train_loader, val_loader = get_train_val_data_loaders(dataset, batch_size=batch_size)

    model = USADModel(input_length=input_length, hidden_size=hidden_size)

    model = fit_with_early_stopping(train_loader, val_loader, model, device,
                                    patience=patience, num_epochs=num_epochs,
                                    learning_rate=learning_rate,
                                    writer=writer, verbose=False)

    dataset.set_flag('test')
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    anomaly_score, labels_list = predict_test_scores(model, test_dataloader, device,
                                                    alpha=.5, beta=.5)
    compute_results_csv(anomaly_score, labels_list, win_size, save_dir)

    # Save the models
    torch.save(model.state_dict(), os.path.join(out_path, 'model'))

    # Save hyperparameters
    init_params = {'input_length': input_length,
                    'hidden_size': hidden_size,
                    }
    algo_config_filename = os.path.join(out_path, "init_params")
    with open(algo_config_filename, "wb") as file:
        pickle.dump(init_params, file)

def load_model(save_dir, device):
    """Load model.

    Args:
        save_dir (str): Path to the folder where outputs are stored.
        device (torch.device): Device

    Returns:
        nn.Module: Model
    """

    out_path = os.path.join(save_dir, 'models')
    algo_config_filename = os.path.join(out_path, "init_params")
    with open(os.path.join(algo_config_filename), "rb") as file:
        init_params = pickle.load(file)

    # init params must contain only arguments of algo_class's constructor
    model = USADModel(input_length=init_params['input_length'],
                    hidden_size=init_params['hidden_size'])
    #device = algo.device
    model_path = os.path.join(out_path, 'model')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    return model
