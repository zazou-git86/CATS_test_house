"""
Copyright (c) 2024 Orange - All rights reserved

Author:  Joël Roman Ky
This code is distributed under the terms and conditions
of the MIT License (https://opensource.org/licenses/MIT)
"""

import logging
import sys
import pickle
import os

from tqdm import trange
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from utils.algorithm_utils import AverageMeter, get_train_val_data_loaders
from utils.evaluation_utils import compute_results_csv



# Get logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class AutoEncoderModel(nn.Module):
    """AutoEncoder model.
    """
    def __init__(self, input_length: int,
                hidden_size: int):
        """Auto-Encoder model architecture.

        Args:
            input_length (int)          : The number of input features.
            hidden_size (int)           : The hidden size.
            seed (int)                  : The random generator seed.
            gpu (int)                   : The number of the GPU device.
        """
        # Each point is a flattened window and thus has as many features
        # as sequence_length * features
        super().__init__()
        # input_length = n_features * sequence_length

        # creates powers of two between eight and the next smaller power from the input_length
        dec_steps = 2 ** np.arange(max(np.ceil(np.log2(hidden_size)), 2),
                                    np.log2(input_length))[1:]
        dec_setup = np.concatenate([[hidden_size], dec_steps.repeat(2),
                                    [input_length]])
        enc_setup = dec_setup[::-1]

        layers = np.array([[nn.Linear(int(a), int(b)), nn.Tanh()]
                        for a, b in enc_setup.reshape(-1, 2)]).flatten()[:-1]
        self._encoder = nn.Sequential(*layers)
        # self.to_device(self._encoder)

        layers = np.array([[nn.Linear(int(a), int(b)), nn.Tanh()]
                        for a, b in dec_setup.reshape(-1, 2)]).flatten()[:-1]
        self._decoder = nn.Sequential(*layers)
        # self.to_device(self._decoder)

    def forward(self, ts_batch, return_latent: bool=False):
        """Forward function of the Auto-Encoder.

        Args:
            ts_batch        : The batch input.
            return_latent   : If the latent vector must be returned. 
                            Defaults to False.

        Returns:
                The reconstructed batch.
        """
        #flattened_sequence = ts_batch.view(ts_batch.size(0), -1)
        enc = self._encoder(ts_batch)
        dec = self._decoder(enc)
        reconstructed_sequence = dec.view(ts_batch.size())
        return (reconstructed_sequence, enc) if return_latent else reconstructed_sequence

def fit_with_early_stopping(train_loader, val_loader, model, device, patience,
                            num_epochs, learning_rate,
                            writer, verbose=True):
    """The fitting function of the Auto Encoder.

    Args:
        train_loader (Dataloader)   : The train dataloader.
        val_loader (Dataloader)     : The val dataloader.
        model (nn.Module)           : The Pytorch model.
        patience (int)              : The number of epochs to wait for early stopping.
        num_epochs (int)            : The max number of epochs.
        learning_rate (float)       : The learning rate.
        writer (SummaryWriter)      : The Tensorboard Summary Writer.
        verbose (bool, optional)    : Defaults to True.

    Returns:
                        [nn.Module ]: The fitted model.
    """
    model.to(device)  # .double()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
            train_loss = train(train_loader, model, optimizer, device)


            # Get Validation loss
            #logger.debug("Begin evaluation")
            val_loss = validation(val_loader, model, device)

            if verbose:
                # logger.info(f"Epoch: [{epoch+1}/{num_epochs}] - Train loss: {train_loss:.2f} - \
                #             Val loss: {val_loss:.2f}")
                logger.info("Epoch: [%d/%d] - Train loss: %2f - Val loss: %2f",
                            epoch+1, num_epochs, train_loss, val_loss)

            # Write in TensorBoard
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('val_loss', val_loss, epoch)

            # Check if the validation loss improve or not
            if val_loss < best_val_loss :
                best_val_loss = val_loss
                epoch_wo_improv = 0
                best_params = model.state_dict()
            elif val_loss >= best_val_loss:
                epoch_wo_improv += 1

        else:
            # No improvement => early stopping is applied and best model is kept
            model.load_state_dict(best_params)
            break
    return model


def train(train_loader, model, optimizer, device):
    """The training step.

    Args:
        train_loader (Dataloader)       : The train data loader.
        model (nn.Module)               : The Pytorch model.
        optimizer (torch.optim)         : The Optimizer.
        epoch (int)                     : The max number of epochs.

    Returns:
                The average loss on the epoch.
    """
    # Compute statistics
    loss_meter = AverageMeter()

    model.train()
    for ts_batch in train_loader:
        ts_batch = ts_batch.to(device)
        output, latent = model(ts_batch, True)
        loss = nn.MSELoss(reduction="mean")(output, ts_batch)
        loss += 0.5*latent.norm(2, dim=1).mean()
        model.zero_grad()
        loss.backward()
        optimizer.step()
        # multiplying by length of batch to correct accounting for incomplete batches
        loss_meter.update(loss.item())
        #train_loss.append(loss.item()*len(ts_batch))

    #train_loss = np.mean(train_loss)/train_loader.batch_size
    #train_loss_by_epoch.append(loss_meter.avg)

    return loss_meter.avg

def validation(val_loader, model, device):
    """The validation step.

    Args:
        val_loader (Dataloader)         : The val data loader.
        model (nn.Module)               : The Pytorch model.
        optimizer (torch.optim)         : The Optimizer.
        epoch (int)                     : The max number of epochs.

    Returns:
                The average loss on the epoch.
    """

    # Compute statistics
    loss_meter = AverageMeter()

    model.eval()
    #val_loss = []
    with torch.no_grad():
        for ts_batch in val_loader:
            ts_batch = ts_batch.to(device)
            output, latent = model(ts_batch, True)
            loss = nn.MSELoss(reduction="mean")(output, ts_batch)
            loss += 0.5*latent.norm(2, dim=1).mean()
            #val_loss.append(loss.item()*len(ts_batch))
            loss_meter.update(loss.item())
        return loss_meter.avg

@torch.no_grad()
def predict_test_scores(model, test_loader, device):
    """The prediction step.

    Args:
        model (nn.Module)               : The PyTorch model.
        test_loader (Dataloader)        : The test dataloader.
        latent (bool, optional)         : If latent variable is used. Defaults to False.

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
        #ts_batch = ts_batch.float().to(model.device)

        #output = model(ts_batch)[:, -1]
        output = model(ts_batch)
        #error = nn.L1Loss(reduction='none')(output, ts_batch)
        error = torch.linalg.norm(output-ts_batch, ord=1, dim=-1)
        reconstr_scores.append(error.cpu().numpy())
        labels_list.append(label.cpu().numpy())
    reconstr_scores = np.concatenate(reconstr_scores)
    labels_list = np.concatenate(labels_list, axis=0)
    print(reconstr_scores.shape)
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
    input_length = feat_size#feat_size*win_size

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

    model = AutoEncoderModel(input_length=input_length, hidden_size=hidden_size)

    model = fit_with_early_stopping(train_loader, val_loader, model, device, patience=patience,
                                        num_epochs=num_epochs, learning_rate=learning_rate,
                                        writer=writer, verbose=False)

    dataset.set_flag('test')
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    anomaly_score, labels_list = predict_test_scores(model, test_dataloader, device)
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
    model = AutoEncoderModel(input_length=init_params['input_length'],
                    hidden_size=init_params['hidden_size'])
    #device = algo.device
    model_path = os.path.join(out_path, 'model')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    return model
