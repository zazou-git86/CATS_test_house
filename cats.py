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
from losses.gcl import GCLoss
from losses.dtw_loss import DTWLoss
from losses.tcl import TCLoss
from encoders.ts2vec import TSEncoder
from encoders.tcn_encoder import TemporalConvNet
from utils.algorithm_utils import AverageMeter, get_train_val_data_loaders
from utils.evaluation_utils import compute_results_csv

# Get logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

class CATS(nn.Module):
    """CATS Architecture module.
    """
    def __init__(self, input_size:int, proj_size:int, win_size:int,
                output_size:int, encoder_type:str='ts2vec', num_layers:int=1):
        """CATS Pytorch module/

        Args:
            input_size (int): Input size n_feat for a batch of size (batch_size, time_step, n_feat)
            proj_size (int): Projection head dimension
            win_size (int): Time series window size
            output_size (int): Embedding size
            encoder_type (str, optional): Type of encoder to use. Defaults to 'ts2vec'.
            num_layers (int, optional): Number of layers of LSTM encoder. Defaults to 1.

        Raises:
            ValueError: _description_
        """
        super(CATS, self).__init__()
        self.encoder_type = encoder_type
        if self.encoder_type == 'lstm':
            self.base = nn.LSTM(input_size,
                                hidden_size=output_size,
                                num_layers=num_layers)
        elif self.encoder_type == 'tcn':
            num_channels = [output_size] * 10
            self.base = TemporalConvNet(input_size, num_channels,
                                        kernel_size=3,
                                        dropout=.2)
        elif self.encoder_type == 'ts2vec':
            self.base = TSEncoder(input_dims=input_size, output_dims=output_size)
        else:
            raise ValueError('Encoder type not implemented')
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
        self.proj_size = proj_size


    def forward(self, x, mask=False):
        """Forward pass

        Args:
            x (torch.Tensor): Batch tensor.
            mask (bool, optional): Apply ts2vec mask. Defaults to False.

        Returns:
            (h_i, v_i): latent vectors, projection vectors.
        """
        if self.encoder_type == 'lstm':
            h_i, _ = self.base(x)
        elif self.encoder_type == 'tcn':
            out = x.permute(0, 2, 1)
            z_i = self.base(out)
            h_i = z_i.permute(0, 2, 1)
        elif self.encoder_type == 'ts2vec':
            h_i = self.base(x, mask)
        out = h_i.reshape(h_i.size(0), -1)
        #print(out.shape)
        #v_i = self.projection(out)
        v_i = self.projection(out)
        return h_i, v_i

def train(train_loader, model, loss_cons_fn, optimizer, device, scheduler,
        loss_temp_fn=None, tcl_clust=False, gcl_clust=False, tcl_inst=False,
        gcl_inst=False, coef_1 =.5, coef_2=.5, crop=True):
    """Training step.

    Args:
        train_loader (torch.Dataloader): Training dataloader.
        model (nn.Module): CATS model.
        loss_cons_fn (GCL): GCL loss function.
        optimizer (torch.optim): Optimizer.
        device (torch.device): Device.
        scheduler ( torch.optim.lr_scheduler): Scheduler.
        loss_temp_fn (TCL, optional): TCL loss function. Defaults to None.
        tcl_inst (bool, optional): Apply TCL. Defaults to False.
        gcl_inst (bool, optional): Apply GCL. Defaults to False.
        coef_1 (float, optional): GCL coefficient weight. Defaults to .5.
        coef_2 (float, optional): TCL coefficient weight. Defaults to .5.
        crop (bool, optional): Apply random cropping. Defaults to True.

    Returns:
        (*torch.Tensor): Loss values.
    """
    loss_cons_meter = AverageMeter()
    loss_cons_clust_meter = AverageMeter()
    loss_temp_meter = AverageMeter()
    loss_temp_clust_meter = AverageMeter()
    total_loss_meter = AverageMeter()

    model.train()
    for batch in train_loader:
        pos_1, pos_2, neg = batch
        pos_1, pos_2 = pos_1.to(device), pos_2.to(device)
        neg = [neg_i.to(device) for neg_i in neg]

        optimizer.zero_grad()

        pos_feat_1, pos_embedding_1 = model(pos_1, mask=False)
        pos_feat_2, pos_embedding_2 = model(pos_2, mask=False)
        neg_outs = [model(neg_i, mask=False) for neg_i in neg]
        neg_embedding = [neg_i[1] for neg_i in neg_outs]
        neg_feat = [neg_i[0] for neg_i in neg_outs]

        loss_cons = 0.
        loss_temp = 0.
        loss_cons_clust = 0.
        loss_temp_clust = 0.

        if loss_cons_fn is not None:
            if gcl_inst:

                loss_cons = loss_cons_fn(pos_embedding_1, pos_embedding_2,
                                        neg_embedding, cluster=False)
                loss_cons_meter.update(loss_cons.item())
            # if gcl_clust:
            #     loss_cons_clust = loss_cons_fn(prob_pos_1, prob_pos_2, neg_prob, cluster=True)
            #     loss_cons_clust_meter.update(loss_cons_clust.item())
            loss_cons = loss_cons if (not gcl_clust) \
                else (loss_cons_clust if (not gcl_inst) else (loss_cons + loss_cons_clust) / 2)

        if loss_temp_fn is not None:
            # Compute the temporal contrastive loss
            # Udate margin for triplet loss
            #update = epoch % update_each == 0
            update = False
            if tcl_clust:
                loss_temp_clust = loss_temp_fn(pos_feat_1, pos_feat_2, neg_feat, cluster=True)
                loss_temp_clust_meter.update(loss_temp_clust.item())

            if tcl_inst:
                loss_temp = loss_temp_fn(pos_feat_1, pos_feat_2, neg_feat,
                                        update=update, cluster=False, crop=crop)
                loss_temp_meter.update(loss_temp.item())

            loss_temp = loss_temp if (not tcl_clust) \
                else (loss_temp_clust if (not tcl_inst) else (loss_temp + loss_temp_clust) / 2)

        # Compute the total loss with EMA
        #update_ema_params = (ema_params is not None)
        loss = coef_1 * loss_cons + coef_2 * loss_temp
        # Compute the grads
        loss.backward()
        total_loss_meter.update(loss.item())

        optimizer.step()
    if scheduler is not None:
        scheduler.step()

    #if loss_svd_fn is not None:
    return loss_cons_meter.avg, loss_temp_meter.avg, total_loss_meter.avg,\
            loss_cons_clust_meter.avg, loss_temp_clust_meter.avg
    # else:
    #     return loss_cons_meter.avg, -1000

def validation(val_loader, model, device, loss_cons_fn,
                loss_temp_fn=None, tcl_clust=False, gcl_clust=False,
                tcl_inst=False, gcl_inst=False,
                coef_1 = .5, coef_2=.5, crop=True):
    """Validation step.

    Args:
        val_loader (torch.Dataloader): Validation dataloader.
        model (nn.Module): CATS model.
        device (torch.device): Device.
        loss_cons_fn (GCL): GCL loss function.
        loss_temp_fn (TCL, optional): TCL loss function. Defaults to None.
        tcl_inst (bool, optional): Apply TCL. Defaults to False.
        gcl_inst (bool, optional): Apply GCL. Defaults to False.
        coef_1 (float, optional): GCL coefficient weight. Defaults to .5.
        coef_2 (float, optional): TCL coefficient weight. Defaults to .5.
        crop (bool, optional): Apply random cropping. Defaults to True.

    Returns:
        (*torch.Tensor): Loss values.
    """
    loss_cons_meter = AverageMeter()
    loss_cons_clust_meter = AverageMeter()
    loss_temp_meter = AverageMeter()
    loss_temp_clust_meter = AverageMeter()
    total_loss_meter = AverageMeter()

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            pos_1, pos_2, neg = batch
            pos_1, pos_2 = pos_1.to(device), pos_2.to(device)
            neg = [neg_i.to(device) for neg_i in neg]

            pos_feat_1, pos_embedding_1 = model(pos_1)
            pos_feat_2, pos_embedding_2 = model(pos_2)
            neg_outs = [model(neg_i) for neg_i in neg]
            neg_embedding = [neg_i[1] for neg_i in neg_outs]
            neg_feat = [neg_i[0] for neg_i in neg_outs]


            loss_cons = 0.
            loss_cons_clust = 0.
            loss_temp = 0.
            loss_temp_clust = 0.

            if loss_cons_fn is not None:
                if gcl_inst:
                    loss_cons = loss_cons_fn(pos_embedding_1, pos_embedding_2,
                                            neg_embedding, cluster=False)
                    loss_cons_meter.update(loss_cons.item())
                # if gcl_clust:
                #     loss_cons_clust = loss_cons_fn(prob_pos_1, prob_pos_2, neg_prob, cluster=True)
                #     loss_cons_clust_meter.update(loss_cons_clust.item())
                    #loss_cons = (loss_cons + loss_cons_clust) / 2
                loss_cons = loss_cons if (not gcl_clust) \
                    else (loss_cons_clust if (not gcl_inst) else (loss_cons + loss_cons_clust) / 2)

            if loss_temp_fn is not None:
                # Compute the temporal contrastive loss
                if tcl_clust:
                    loss_temp_clust = loss_temp_fn(pos_feat_1, pos_feat_2, neg_feat, cluster=True)
                    loss_temp_clust_meter.update(loss_temp_clust.item())
                if tcl_inst:
                    loss_temp = loss_temp_fn(pos_feat_1, pos_feat_2,
                                            neg_feat, cluster=False, crop=crop)
                    # #loss_temp = loss_temp_fn(pos_pred_1, pos_pred_2, neg_pred)
                    loss_temp_meter.update(loss_temp.item())

                loss_temp = loss_temp if (not tcl_clust) \
                    else (loss_temp_clust if (not tcl_inst) else (loss_temp + loss_temp_clust) / 2)
                #loss_temp = loss_temp_clust

            # Compute the total loss with EMA
            loss = coef_1 * loss_cons + coef_2 * loss_temp
            total_loss_meter.update(loss.item())

    return loss_cons_meter.avg, loss_temp_meter.avg, total_loss_meter.avg, \
            loss_cons_clust_meter.avg, loss_temp_clust_meter.avg

def fit_with_early_stopping(train_loader, val_loader, loss_cons_fn, loss_temp_fn,
                            model, patience, num_epochs, learning_rate,
                            writer, device, weight_decay=1e-5,
                            verbose=True, coef_1=.5, coef_2=.3,
                            tcl_clust=False, gcl_clust=False, tcl_inst=False, gcl_inst=False,
                            crop=True):
    """Train and validation with early stopping.

    Args:
        train_loader (torch.Dataloader): Train dataloader.
        val_loader (torch.Dataloader): Validation dataloader.
        loss_cons_fn (GCL): GCL loss function.
        loss_temp_fn (TCL): TCL loss function.
        model (nn.Module): CATS model.
        patience (int): Number of epochs to wait before stopping.
        num_epochs (int): Number of epochs for training.
        learning_rate (float): Learning rate.
        writer (_type_): Tensorboard summary writer.
        device (torch.device): Device.
        weight_decay (float, optional): Weight decay. Defaults to 1e-5.
        update_each (int, optional): Frequency to update TCL margin. Defaults to 5.
        verbose (bool, optional): If print training information. Defaults to True.
        coef_1 (float, optional): GCL weight coefficient. Defaults to .5.
        coef_2 (float, optional): TCL weight coefficient. Defaults to .3.
        tcl_inst (bool, optional): Apply TCL. Defaults to False.
        gcl_inst (bool, optional): Apply GCL. Defaults to False.
        crop (bool, optional): Apply random cropping. Defaults to True.

    Returns:
        _type_: _description_
    """
    # model.to(model.device)  # .double()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10,
                                                                    T_mult=1, eta_min=1e-6)
    #scheduler = None


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
                logger.info('Epoch %d/%d.', epoch + 1, num_epochs)

            # Compute k_means centroids using soft dtw
            # if (epoch % update_each == 0):
            #     if loss_temp_fn is not None:
            #         _ = loss_temp_fn.compute_k_means(model, train_loader, num_clusters=2)
            #         print("K-means centroids computed !")
            #         tcl_clust = True
            #         tcl_inst = False

            # Train the model
            #logger.debug("Begin training...")
            train_cons_loss, train_temp_loss, \
                train_loss, train_cons_clust, train_temp_clust = train(train_loader,
                                                        model,
                                                        loss_cons_fn,
                                                        optimizer,
                                                        device,
                                                        scheduler,
                                                        loss_temp_fn=loss_temp_fn,
                                                        tcl_clust=tcl_clust, gcl_clust=gcl_clust,
                                                        tcl_inst=tcl_inst, gcl_inst=gcl_inst,
                                                        coef_1=coef_1, coef_2=coef_2, crop=crop)


            # Get Validation loss
            #logger.debug("Begin evaluation")
            val_cons_loss, val_temp_loss, \
                val_loss, val_cons_clust, val_temp_clust = validation(val_loader,
                                                        model,
                                                        device,
                                                        loss_cons_fn,
                                                        loss_temp_fn=loss_temp_fn,
                                                        tcl_clust=tcl_clust, gcl_clust=gcl_clust,
                                                        tcl_inst=tcl_inst, gcl_inst=gcl_inst,
                                                        coef_1=coef_1, coef_2=coef_2, crop=crop)


            if verbose:
                logger.info("Epoch: [%d/%d] - \
                            Train cons loss: %2f / Train temp loss: %2f - \
                            Val cons loss: %2f / Val temp loss: %2f",
                            epoch+1, num_epochs,
                            train_cons_loss, train_temp_loss,
                            val_cons_loss, val_temp_loss)

            # Write in TensorBoard
            writer.add_scalar('cons_loss/train', train_cons_loss, epoch)
            writer.add_scalar('cons_loss/val', val_cons_loss, epoch)
            writer.add_scalar('cons_clust/train', train_cons_clust, epoch)
            writer.add_scalar('cons_clust/val', val_cons_clust, epoch)
            writer.add_scalar('temp_loss/train', train_temp_loss, epoch)
            writer.add_scalar('temp_loss/val', val_temp_loss, epoch)
            writer.add_scalar('temp_clust/train', train_temp_clust, epoch)
            writer.add_scalar('temp_clust/val', val_temp_clust, epoch)
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

def predict_test_scores(model, test_dataloader, device, loss_svd_fn, latent=False,
        test=True, cosine=False):
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
    embeddings_neg = []
    labels_list = []
    anomaly_score = []

    with torch.no_grad():
        for data in test_dataloader:
            pos_1, pos_2, negative, label = data
            pos_1, pos_2, label = pos_1.to(device), pos_2.to(device), label.to(device)
            negative = [neg_i.to(device) for neg_i in negative]

            pos_feat_1, pos_1_embedding = model(pos_1)
            _, pos_2_embedding = model(pos_2)
            neg_outs = [model(neg_i) for neg_i in negative]
            negative_embedding = [neg_i[1] for neg_i in neg_outs]

            loss = loss_svd_fn(pos_feat_1, test=test, cosine=cosine)

            anomaly_score.append(loss.cpu().numpy())
            labels_list.append(label.cpu().numpy())
            if latent:
                embeddings_pos_1.append(pos_1_embedding.cpu().numpy())
                embeddings_pos_2.append(pos_2_embedding.cpu().numpy())
                if negative_embedding is not None:
                    embeddings_neg.append(negative_embedding.cpu().numpy())

    anomaly_score = np.concatenate(anomaly_score, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    if latent:
        embeddings_pos_1 = np.concatenate(embeddings_pos_1, axis=0)
        embeddings_pos_2 = np.concatenate(embeddings_pos_2, axis=0)
        if negative_embedding is not None:
            embeddings_neg = np.concatenate(embeddings_neg, axis=0)

    return anomaly_score, labels_list, embeddings_pos_1, embeddings_pos_2, embeddings_neg

def eval_on_data(dataset, save_dir, output_size, proj_size,
                batch_size=512, learning_rate=.001,
                num_epochs=100, patience=50, min_crop_ratio=.9, max_crop_ratio=1,
                use_gpu=True, tcl_clust=False, gcl_clust=False,
                tcl_inst=True, gcl_inst=True, crop=True,
                temperature=.1, gamma=1, margin=5):
    """Train and evaluate CATS on a dataset.

    Args:
        dataset (torch.Dataset): Dataset.
        save_dir (str): Path to save outputs.
        output_size (int): Embedding size.
        proj_size (int): Projection size.
        batch_size (int, optional): Batch size. Defaults to 512.
        learning_rate (float, optional): Learning rate. Defaults to .001.
        num_epochs (int, optional): Max number of epochs. Defaults to 100.
        patience (int, optional): Number of epochs to wait before stopping. Defaults to 50.
        min_crop_ratio (float, optional): Min random cropping. Defaults to .9.
        max_crop_ratio (int, optional): Max random cropping. Defaults to 1.
        use_gpu (bool, optional): Use GPU. Defaults to True.
        tcl_inst (bool, optional): Apply TCL. Defaults to True.
        gcl_inst (bool, optional): Apply GCL. Defaults to True.
        crop (bool, optional): Apply random cropping. Defaults to True.
        temperature (float, optional): GCL temperature. Defaults to .1.
        gamma (int, optional): TCL gamma. Defaults to 1.
        margin (int, optional): TCL margin. Defaults to 5.
    """

    win_size = dataset.win_size #data_dict['train']['x'].shape[1]
    feat_size = dataset.n_features #data_dict['train']['x'].shape[2]
    input_size = feat_size
    crop_size_min = int(min_crop_ratio*win_size)
    crop_size_max = int(max_crop_ratio*win_size)+1

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

    encoder_type = 'ts2vec'
    model = CATS(input_size=input_size,
                proj_size=proj_size,
                win_size=win_size,
                output_size=output_size,
                encoder_type=encoder_type)

    loss_cons_fn = GCLoss(device=device,
                        temperature=temperature)
    temp_cons_loss = DTWLoss(device, use_soft_dtw=True, use_cuda=False,
                            gamma=gamma)
    loss_temp_fn = TCLoss(loss_fn=temp_cons_loss,
                            device=device,
                            crop_size_min=crop_size_min,
                            crop_size_max=crop_size_max,
                            if_use_dtw=True,
                            margin=margin,
                            )
    loss_svd_fn = SVDLoss(radius=0., device=device)


    model = fit_with_early_stopping(train_loader, val_loader, loss_cons_fn, loss_temp_fn,
                            model, patience, num_epochs, learning_rate,
                            writer, device, verbose=False, coef_1=.5, coef_2=.5,
                            tcl_clust=tcl_clust, gcl_clust=gcl_clust,
                            tcl_inst=tcl_inst, gcl_inst=gcl_inst, crop=crop)
    _ = loss_svd_fn.init_center(model, train_loader, win_size, output_size, eps=.1)

    dataset.set_flag('test')
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    anomaly_score, labels_list, _, _ , _= predict_test_scores(model, test_dataloader, device,
                                                        loss_svd_fn, latent=False,
                                                        test=True, cosine=False)

    compute_results_csv(anomaly_score, labels_list, win_size, save_dir)

    # Save the models
    torch.save(model.state_dict(), os.path.join(out_path, 'model'))

    # Save hyperparameters
    init_params = {'center': loss_svd_fn.center,
                    'radius': loss_svd_fn.radius,
                    'input_size': feat_size,
                    'output_size': output_size,
                    'proj_size': proj_size,
                    'win_size': win_size,
                    'encoder_type': encoder_type,
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

    model = CATS(
                    input_size=init_params['input_size'],
                    proj_size=init_params['proj_size'],
                    win_size=init_params['win_size'],
                    output_size=init_params['output_size'],
                    encoder_type=init_params['ecoder_type'])

    #device = algo.device
    model_path = os.path.join(out_path, 'model')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Load SVDD center and radius
    center = init_params['center']
    radius = init_params['radius']
    loss_svd_fn = SVDLoss(radius=radius, device=device, center=center)
    return model, loss_svd_fn
