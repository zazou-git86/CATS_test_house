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
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from losses.svd_loss import SVDLoss
from encoders.ts2vec import TSEncoder
from utils.algorithm_utils import AverageMeter, get_train_val_data_loaders
from utils.evaluation_utils import compute_results_csv

# Get logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

class TS2Vec(nn.Module):
    """TS2Vec implementation for anomaly detection
    based on https://github.com/zhihanyue/ts2vec/tree/main.
    """
    def __init__(self, input_size, output_size):
        """
        Args:
            input_size (int): Input size.
            output_size (int): Embedding size.
        """
        super().__init__()
        self.base = TSEncoder(input_dims=input_size, output_dims=output_size)

    def forward(self, x, mask=False):
        """Forward pass.

        Args:
            x (torch.Tensor): Input batch tensor.
            mask (bool, optional): Apply random masking. Defaults to False.

        Returns:
            torch.Tensor: Model outputs.
        """
        feat = self.base(x, mask=mask)
        return feat

def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]

def hierarchical_contrastive_loss(z1, z2, temporal_unit=0):
    """Hierarchical contrastive loss.

    Args:
        z1 (torch.Tensor): Cropped tensors batch.
        z2 (torch.Tensor): Cropped tensors batch.
        temporal_unit (int, optional): _description_. Defaults to 0.

    Returns:    
        torch.Tensor: Loss tensor.
    """
    loss_temp = torch.tensor(0., device=z1.device)
    loss_cons = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        loss_cons += instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            loss_temp += temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        loss_cons += instance_contrastive_loss(z1, z2)
        d += 1
    loss_cons /= d
    loss_temp /= d
    return loss_cons, loss_temp

def instance_contrastive_loss(z1, z2):
    """Instance contrastive loss.

    Args:
        z1 (torch.Tensor): Cropped tensors batch.
        z2 (torch.Tensor): Cropped tensors batch.

    Returns:    
        torch.Tensor: Loss tensor.
    """
    B, _ = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss

def temporal_contrastive_loss(z1, z2):
    """Temporal contrastive loss.

    Args:
        z1 (torch.Tensor): Cropped tensors batch.
        z2 (torch.Tensor): Cropped tensors batch.

    Returns:    
        torch.Tensor: Loss tensor.
    """
    _, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss

def train(train_loader, model, optimizer, device, scheduler,
        coef_1=.5, coef_2=.5):
    """Training step.

    Args:
        train_loader (torch.Dataloader): Training dataloader.
        model (nn.Module): CATS model.
        optimizer (torch.optim): Optimizer.
        device (torch.device): Device.
        scheduler (torch.optim.lr_scheduler): Scheduler.
        coef_1 (float): Instance loss coefficient weight. Defaults to .5.
        coef_2 (float): Temporal loss coefficient weight. Defaults to .5.

    Returns:
        (torch.Tensor*): Loss tensor.
    """
    loss_cons_meter = AverageMeter()
    loss_temp_meter = AverageMeter()
    total_loss_meter = AverageMeter()

    model.train()

    for batch in train_loader:
        pos_1, _, _ = batch
        pos_1 = pos_1.to(device)
        optimizer.zero_grad()

        ts_l = pos_1.size(1)
        crop_l = np.random.randint(low=2 ** 1, high=ts_l+1)
        crop_left = np.random.randint(ts_l - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
        crop_offset = np.random.randint(low=-crop_eleft,
                                        high=ts_l - crop_eright + 1,
                                        size=pos_1.size(0))
        view_left = take_per_row(pos_1, crop_offset + crop_eleft, crop_right - crop_eleft)
        view_right = take_per_row(pos_1, crop_offset + crop_left, crop_eright - crop_left)

        optimizer.zero_grad()

        out1 = model(view_left, mask=True)
        out1 = out1[:, -crop_l:]

        out2 = model(view_right, mask=True)
        out2 = out2[:, :crop_l]

        loss_cons, loss_temp = hierarchical_contrastive_loss(out1, out2)
        loss_cons_meter.update(loss_cons.item())
        loss_temp_meter.update(loss_temp.item())

        # Compute the total loss with EMA
        loss = coef_1 * loss_cons + coef_2 * loss_temp

        # Compute the grads
        loss.backward()
        total_loss_meter.update(loss.item())

        optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return loss_cons_meter.avg, loss_temp_meter.avg, total_loss_meter.avg

def validation(val_loader, model, device, coef_1=.5, coef_2=.5):
    """Validation step.

    Args:
        val_loader (torch.Dataloader): Validation dataloader.
        model (nn.Module): CATS model.
        device (torch.device): Device.
        coef_1 (float): Instance loss coefficient weight. Defaults to .5.
        coef_2 (float): Temporal loss coefficient weight. Defaults to .5.

    Returns:
        (torch.Tensor*): Loss tensor.
    """
    loss_cons_meter = AverageMeter()
    loss_temp_meter = AverageMeter()
    total_loss_meter = AverageMeter()

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            pos_1, _, _ = batch
            pos_1 = pos_1.to(device)

            ts_l = pos_1.size(1)
            crop_l = np.random.randint(low=2 ** 1, high=ts_l+1)
            crop_left = np.random.randint(ts_l - crop_l + 1)
            crop_right = crop_left + crop_l
            crop_eleft = np.random.randint(crop_left + 1)
            crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
            crop_offset = np.random.randint(low=-crop_eleft,
                                            high=ts_l - crop_eright + 1,
                                            size=pos_1.size(0))
            view_left = take_per_row(pos_1, crop_offset + crop_eleft, crop_right - crop_eleft)
            view_right = take_per_row(pos_1, crop_offset + crop_left, crop_eright - crop_left)

            out1 = model(view_left)
            out1 = out1[:, -crop_l:]

            out2 = model(view_right)
            out2 = out2[:, :crop_l]


            loss_cons, loss_temp = hierarchical_contrastive_loss(out1, out2)
            loss_cons_meter.update(loss_cons.item())
            loss_temp_meter.update(loss_temp.item())

            # Compute the total loss with EMA
            loss = coef_1 * loss_cons + coef_2 * loss_temp

            # Compute the grads
            total_loss_meter.update(loss.item())

    return loss_cons_meter.avg, loss_temp_meter.avg, total_loss_meter.avg

def fit_with_early_stopping(train_loader, val_loader,
                            model, patience, num_epochs, learning_rate,
                            writer, device, weight_decay=1e-5,
                            verbose=True, coef_1=.5, coef_2=.5):
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
        coef_1 (float): Instance loss coefficient weight. Defaults to .5.
        coef_2 (float): Temporal loss coefficient weight. Defaults to .5.

    Returns:
        torch.Module: Model.
    """
    # model.to(model.device)  # .double()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10,
                                                                    T_mult=1, eta_min=1e-6)


    model = model.to(device)
    model.train()

    best_val_loss = np.inf
    epoch_wo_improv = 0
    best_params = model.state_dict()



    # assuming first batch is complete
    for epoch in trange(num_epochs):
        # If improvement continue training
        if epoch_wo_improv < patience:
            # logging.debug(f'Epoch {epoch + 1}/{num_epochs}.')
            if verbose:
                logger.debug('Epoch %d/%d.', epoch + 1, num_epochs)


            # Train the model
            #logger.debug("Begin training...")
            train_cons_loss, train_temp_loss, train_loss = train(train_loader,
                                                            model,
                                                            optimizer,
                                                            device,
                                                            scheduler,
                                                            coef_1=coef_1, coef_2=coef_2)


            # Get Validation loss
            #logger.debug("Begin evaluation")
            val_cons_loss, val_temp_loss, val_loss = validation(val_loader,
                                                        model,
                                                        device,
                                                        coef_1=coef_1, coef_2=coef_2)

            if verbose:
                logger.info("Epoch: [%d/%d] - \
                            Train cons loss: %2f / Train temp loss: %2f / - \
                            Val cons loss: %2f  Val temp loss: %2f",
                            epoch+1, num_epochs,
                            train_cons_loss, train_temp_loss,
                            val_cons_loss, val_temp_loss)

            # Write in TensorBoard
            writer.add_scalar('cons_loss/train', train_cons_loss, epoch)
            writer.add_scalar('cons_loss/val', val_cons_loss, epoch)
            writer.add_scalar('temp_loss/train', train_temp_loss, epoch)
            writer.add_scalar('temp_loss/val', val_temp_loss, epoch)
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
            pos_1, _, _, label = data
            pos_1, label = pos_1.to(device), label.to(device)

            pos_feat_1 = model(pos_1)
            #pos_2_embedding, _ = model(pos_2)

            loss_svd = loss_svd_fn(pos_feat_1, test=test, cosine=cosine)
            #print(loss_svd.shape)
            #print(loss_svd.shape)
            loss = loss_svd

            anomaly_score.append(loss.cpu().numpy())
            labels_list.append(label.cpu().numpy())
            # if latent:
            #     embeddings_pos_1.append(pos_1_embedding.cpu().numpy())
            #     embeddings_pos_2.append(pos_2_embedding.cpu().numpy())


    anomaly_score = np.concatenate(anomaly_score, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    if latent:
        embeddings_pos_1 = np.concatenate(embeddings_pos_1, axis=0)
        embeddings_pos_2 = np.concatenate(embeddings_pos_2, axis=0)
        return anomaly_score, labels_list, embeddings_pos_1, embeddings_pos_2
    else:
        return anomaly_score, labels_list, embeddings_pos_1, embeddings_pos_2

def eval_on_data(dataset, save_dir, output_size,
                batch_size=512, learning_rate=.001,
                num_epochs=100, patience=50,
                use_gpu=True):
    """Train and evaluate Ts2Vec on a dataset.

    Args:
        dataset (torch.Dataset): Dataset.
        save_dir (str): Path to save outputs.
        output_size (int): Embedding size.
        batch_size (int, optional): Batch size. Defaults to 512.
        learning_rate (float, optional): Learning rate. Defaults to .001.
        num_epochs (int, optional): Max number of epochs. Defaults to 100.
        patience (int, optional): Number of epochs to wait before stopping. Defaults to 50.
        use_gpu (bool, optional): Use GPU. Defaults to True.
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    out_path = os.path.join(save_dir, 'models')
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    win_size = dataset.win_size #data_dict['train']['x'].shape[1]
    feat_size = dataset.n_features #data_dict['train']['x'].shape[2]
    #input_size = feat_size*win_size

    if use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')


    writer = SummaryWriter(log_dir=out_path)


    dataset.set_flag('train')
    train_loader, val_loader = get_train_val_data_loaders(dataset, batch_size=batch_size)

    model = TS2Vec(input_size=feat_size,
                    output_size=output_size)

    loss_svd_fn = SVDLoss(radius=0., device=device)

    model = fit_with_early_stopping(train_loader, val_loader,
                            model, patience, num_epochs, learning_rate,
                            writer, device, weight_decay=1e-5,
                            verbose=False, coef_1=.5, coef_2=.5)
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
                    'output_size': output_size
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
    model = TS2Vec(input_size=init_params['input_size'],
                    output_size=init_params['output_size'])
    #device = algo.device
    model_path = os.path.join(out_path, 'model')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Load SVDD center and radius
    center = init_params['center']
    radius = init_params['radius']
    loss_svd_fn = SVDLoss(radius=radius, device=device, center=center)
    return model, loss_svd_fn
