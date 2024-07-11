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

from tqdm import trange
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from losses.svd_loss import SVDLoss
from encoders.ts2vec import TSEncoder
from utils.algorithm_utils import AverageMeter, get_train_val_data_loaders
from utils.evaluation_utils import compute_results_csv

# Get logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

class SimSiam(nn.Module):
    """SimSiam implementation for AD.
    """
    def __init__(self, input_size, output_size, proj_size, win_size):
        """_summary_

        Args:
            input_size (int): Input size.
            output_size (int): Embedding size.
            proj_size (int): Projection size.
            win_size (int): Window size.
        """
        super().__init__()

        self.base = TSEncoder(input_dims=input_size, output_dims=output_size)

        feat_size = output_size*win_size
        self.win_size = win_size
        self.linear1 = nn.Linear(feat_size, feat_size)
        self.linear2 = nn.Linear(feat_size, feat_size//2)
        self.linear3 = nn.Linear(feat_size//2, proj_size)
        self.relu = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.projection = nn.Sequential(self.linear1,
                                        #nn.BatchNorm1d(feat_size//2),
                                        self.relu,
                                        self.linear2,
                                        self.relu2,
                                        self.linear3,
                                        #nn.BatchNorm1d(proj_size)
                                        )
        self.predictor = nn.Sequential(nn.Linear(proj_size, proj_size//2),
                                        nn.ReLU(),
                                        nn.Linear(proj_size//2, proj_size)
                                        )
        self.proj_size = proj_size


    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input batch tensor.

        Returns:
            torch.Tensor: Model output.
        """
        feat = self.base(x)
        out = feat.reshape(feat.size(0), -1)
        v_i = self.projection(out)
        p_i = self.predictor(v_i)
        return feat, v_i, p_i

def train(train_loader, model, criterion, optimizer, device, scheduler):
    """Training step.

    Args:
        train_loader (torch.Dataloader): Training dataloader.
        model (nn.Module): CATS model.
        criterion (torch.Module): Cosine Similarity loss function.
        optimizer (torch.optim): Optimizer.
        device (torch.device): Device.
        scheduler ( torch.optim.lr_scheduler): Scheduler.

    Returns:
        torch.Tensor: Loss tensor.
    """
    total_loss_meter = AverageMeter()
    model.train()

    for batch in train_loader:
        pos_1, pos_2, _ = batch
        pos_1, pos_2 = pos_1.to(device), pos_2.to(device)
        optimizer.zero_grad()

        _, pos_embedding_1, pos_pred_1 = model(pos_1)
        _, pos_embedding_2, pos_pred_2 = model(pos_2)

        loss = -(criterion(pos_pred_1, pos_embedding_2.detach()).mean() \
                + criterion(pos_pred_2, pos_embedding_1.detach()).mean()) * 0.5

        total_loss_meter.update(loss.item())

        # Compute the grads
        loss.backward()

        optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return total_loss_meter.avg

def validation(val_loader, model, device, criterion):
    """Validation step.

    Args:
        val_loader (torch.Dataloader): Validation dataloader.
        model (nn.Module): CATS model.
        device (torch.device): Device.
        criterion (torch.Module): Cosine Similarity loss function.

    Returns:
        torch.Tensor: Loss tensor.
    """
    total_loss_meter = AverageMeter()
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            pos_1, pos_2, _ = batch
            pos_1, pos_2 = pos_1.to(device), pos_2.to(device)

            _, pos_embedding_1, pos_pred_1 = model(pos_1)
            _, pos_embedding_2, pos_pred_2 = model(pos_2)

            loss = -(criterion(pos_pred_1, pos_embedding_2.detach()).mean() \
                    + criterion(pos_pred_2, pos_embedding_1.detach()).mean()) * 0.5
            total_loss_meter.update(loss.item())


    return total_loss_meter.avg

def fit_with_early_stopping(train_loader, val_loader,
                            model, patience, num_epochs, learning_rate,
                            writer, device, weight_decay=1e-5,
                            verbose=True):
    """Train and validation with early stopping.

    Args:
        train_loader (torch.Dataloader): Train dataloader.
        val_loader (torch.Dataloader): Validation dataloader.
        model (nn.Module): CATS model.
        patience (int): Number of epochs to wait before stopping.
        num_epochs (int): Number of epochs for training.
        learning_rate (float): Learning rate.
        writer (_type_): Tensorboard summary writer.
        device (torch.device): Device.
        weight_decay (float, optional): Weight decay. Defaults to 1e-5.
        verbose (bool, optional): If print training information. Defaults to True.

    Returns:
        torch.Module: Model.
    """
    # model.to(model.device)  # .double()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10,
                                                                    T_mult=1, eta_min=1e-6)
    criterion = nn.CosineSimilarity(dim=1).to(device)

    model = model.to(device)
    model.train()

    best_val_loss = np.inf
    epoch_wo_improv = 0
    best_params = model.state_dict()

    # assuming first batch is complete
    for epoch in trange(num_epochs):
        # If improvement continue training
        if epoch_wo_improv < patience:
            if verbose:
            # logging.debug(f'Epoch {epoch + 1}/{num_epochs}.')
                logger.info('Epoch %d/%d.', epoch + 1, num_epochs)


            # Train the model
            #logger.debug("Begin training...")
            train_loss = train(train_loader,
                            model,
                            criterion,
                            optimizer,
                            device,
                            scheduler)


            # Get Validation loss
            #logger.debug("Begin evaluation")
            val_loss = validation(val_loader,
                                model,
                                device,
                                criterion)

            if verbose:
                logger.info("Epoch: [%d/%d] - \
                            Train cons loss: %2f - Val cons loss: %2f",
                            epoch+1, num_epochs,
                            train_loss, val_loss)

            # Write in TensorBoard
            writer.add_scalar('total_loss/train', train_loss, epoch)
            writer.add_scalar('total_loss/val', val_loss, epoch)

            # Check if the loss is nan
            if np.isnan(train_loss):
                epoch_wo_improv = patience
                print("Training stopped cause nan values.")
                model.load_state_dict(best_params)
            else:
                # Check if the validation loss improve or not
                if val_loss < best_val_loss:
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

def predict_test_scores(model, test_dataloader, device, loss_svd_fn=None,
                        latent=False, test=True, cosine=False):
    """Predict anomaly scores.

    Args:
        model (nn.Module): CATS model.
        test_dataloader (torch.Dataloader): Test dataloader.
        device (torch.device): Device.
        loss_svd_fn (SVDLoss): Function to compute anomaly score
        latent (bool, optional): Return latents vectors. Defaults to False.
        test (bool, optional): Test. Defaults to True.
        cosine (bool, optional): Apply cosine distance. Defaults to False.

    Returns:
        (anomaly_score, labels_list, latent_vectors)
    """

    model.eval()
    embeddings_pos_2 = []
    embeddings_pos_1 = []
    labels_list = []
    anomaly_score = []

    with torch.no_grad():
        for data in test_dataloader:
            pos_1, pos_2, _, label = data
            pos_1, pos_2, label = pos_1.to(device), pos_2.to(device), label.to(device)

            pos_feat_1, pos_1_embedding, _ = model(pos_1)
            _, pos_2_embedding, _ = model(pos_2)

            loss_svd = loss_svd_fn(pos_feat_1, test=test, cosine=cosine)
            #print(loss_svd.shape)
            #print(loss_svd.shape)
            loss = loss_svd

            anomaly_score.append(loss.cpu().numpy())
            labels_list.append(label.cpu().numpy())
            if latent:
                embeddings_pos_1.append(pos_1_embedding.cpu().numpy())
                embeddings_pos_2.append(pos_2_embedding.cpu().numpy())


    anomaly_score = np.concatenate(anomaly_score, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    if latent:
        embeddings_pos_1 = np.concatenate(embeddings_pos_1, axis=0)
        embeddings_pos_2 = np.concatenate(embeddings_pos_2, axis=0)
        return anomaly_score, labels_list, embeddings_pos_1, embeddings_pos_2
    else:
        return anomaly_score, labels_list, embeddings_pos_1, embeddings_pos_2

def eval_on_data(dataset, save_dir, output_size, proj_size,
                batch_size=512, learning_rate=.001,
                num_epochs=100, patience=50,
                use_gpu=True):
    """Train and evaluate SimSiam on a dataset.

    Args:
        dataset (torch.Dataset): Dataset.
        save_dir (str): Path to save outputs.
        output_size (int): Embedding size.
        proj_size (int): Projection size.
        batch_size (int, optional): Batch size. Defaults to 512.
        learning_rate (float, optional): Learning rate. Defaults to .001.
        num_epochs (int, optional): Max number of epochs. Defaults to 100.
        patience (int, optional): Number of epochs to wait before stopping. Defaults to 50.
        use_gpu (bool, optional): Use GPU. Defaults to True.
    """

    win_size = dataset.win_size #data_dict['train']['x'].shape[1]
    feat_size = dataset.n_features #data_dict['train']['x'].shape[2]
    #input_size = feat_size*win_size

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

    model = SimSiam(input_size=feat_size,
                    output_size=output_size,
                    proj_size=proj_size,
                    win_size=win_size)

    loss_svd_fn = SVDLoss(radius=0., device=device)

    model = fit_with_early_stopping(train_loader, val_loader,
                            model, patience, num_epochs, learning_rate,
                            writer, device, weight_decay=1e-5,
                            verbose=False)
    _ = loss_svd_fn.init_center(model, train_loader, win_size, output_size, eps=.1)

    dataset.set_flag('test')
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    anomaly_score, labels_list, _, _ = predict_test_scores(model, test_dataloader,
                                                    device, loss_svd_fn=loss_svd_fn,
                                                    latent=False, test=True,
                                                    cosine=False)

    compute_results_csv(anomaly_score, labels_list, win_size, save_dir)

    # Save the models
    torch.save(model.state_dict(), os.path.join(out_path, 'model'))

    # Save hyperparameters
    init_params = {'center': loss_svd_fn.center,
                    'radius': loss_svd_fn.radius,
                    'input_size': feat_size,
                    'output_size': output_size,
                    'proj_size': proj_size,
                    'win_size': win_size
                    }
    algo_config_filename = os.path.join(out_path, "init_params")
    with open(algo_config_filename, "wb") as file:
        pickle.dump(init_params, file)

def load_model(save_dir, device):
    """Load trained model for inference.

    Args:
        save_dir (str): Path where the outs are stored.
        device (torch.device): Device for loading model.

    Returns:
        (model, anomaly_score_function).
    """

    out_path = os.path.join(save_dir, 'models')
    algo_config_filename = os.path.join(out_path, "init_params")
    with open(os.path.join(algo_config_filename), "rb") as file:
        init_params = pickle.load(file)

    # init params must contain only arguments of algo_class's constructor
    model = SimSiam(input_size=init_params['input_size'],
                    output_size=init_params['output_size'],
                    proj_size=init_params['proj_size'],
                    win_size=init_params['win_size'])
    #device = algo.device
    model_path = os.path.join(out_path, 'model')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Load SVDD center and radius
    center = init_params['center']
    radius = init_params['radius']
    loss_svd_fn = SVDLoss(radius=radius, device=device, center=center)
    return model, loss_svd_fn
