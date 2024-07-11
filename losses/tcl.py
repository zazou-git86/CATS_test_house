"""
Copyright (c) 2024 Orange - All rights reserved

Author:  Joël Roman Ky
This code is distributed under the terms and conditions
of the MIT License (https://opensource.org/licenses/MIT)
"""

import numpy as np
import torch
import torch.nn as nn

from losses.pytorch_kmeans import pairwise_soft_dtw

class TCLoss(nn.Module):
    """Temporal Contrastive Loss.
    """
    def __init__(self, loss_fn, device, crop_size_min=5, crop_size_max=10,
                if_use_dtw=False, max_margin=5, min_margin=1, num_clusters=2,
                temperature=.1, margin=5):
        """_summary_

        Args:
            loss_fn (torch.Module): Similarity loss.
            device (torch.device): Device.
            crop_size_min (int, optional): Min crop size. Defaults to 5.
            crop_size_max (int, optional): Max crop size. Defaults to 10.
            if_use_dtw (bool, optional): Apply DTW similarity. Defaults to False.
            max_margin (int, optional): Max margin value. Defaults to 5.
            min_margin (int, optional): Min margin value. Defaults to 1.
            num_clusters (int, optional): Number of clusters. Defaults to 2.
            temperature (float, optional): Temperature parameter. Defaults to .1.
            margin (int, optional): Margin value. Defaults to 5.
        """
        super(TCLoss, self).__init__()
        self.sim_loss = loss_fn
        self.crop_size_min = crop_size_min
        self.crop_size_max = crop_size_max
        self.use_dtw = if_use_dtw
        self.max_margin = max_margin
        self.min_margin = min_margin
        self.margin = margin
        self.device = device
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.clusters_centers = None
        self.num_clusters = num_clusters

    def _update_margin(self,  new_margin):
        """Get hypersphere radius.
        Args:
            dist (torch.Tensor): _description_
            nu_val (float): Outliers hyperparameters.
        Returns:
            np.array: Radius array.
        """

        self.margin = new_margin# to(self.device)

    def sdtw_similarity(self, vect_x, vect_y):
        """Compute similarity using Soft-DTW divergence.

        Args:
            vect_x (torch.Tensor): Batch 3D tensor (b_size, win_size, n_feats).
            vect_y (torch.Tensor): Batch 3D tensor (b_size, win_size, n_feats).

        Returns:
            torch.Tensor: Similarity matrix (b_size, b_size).
        """
        # Reshape to the good shape
        b_size, win_size, n_feats = vect_x.shape

        vect_x_row = vect_x.unsqueeze(0).expand(b_size, b_size,
                                                win_size, n_feats).reshape(-1, win_size, n_feats)
        vect_y_col = vect_y.unsqueeze(1).expand(b_size, b_size,
                                                win_size, n_feats).reshape(-1, win_size, n_feats)

        sim_matrix = self.sim_loss(vect_x_row, vect_y_col).reshape(b_size, b_size)
        return sim_matrix

    def temporal_contrastive_clustering(self, data1, data2, data3=None):
        """Apply temporal contrastive clustering.

        Args:
            data1 (torch.Tensor): Batch of positive views.
            data2 (torch.Tensor): Batch of positive views
            data3 (torch.Tensor, optional): Batch of negative views. Defaults to None.

        Returns:
            torch.Tensor: Loss value.
        """
        # Concat the data
        batch_data = torch.cat([data1, data2, data3[0]], dim=0)
        b_size = batch_data.shape[0]

        # Compute pairwise distance between batch data and cluster centers
        pair_dist = pairwise_soft_dtw(batch_data, self.clusters_centers, sdtw=self.sim_loss)

        # Get positives distance
        try:
            positives, _ = torch.min(pair_dist, dim=1)
            positives = positives.view(b_size, -1)

            # Get the negatives
            negatives, _ = torch.topk(pair_dist, k=self.num_clusters-1)
            negatives = negatives.view(b_size, -1)
            #negatives = pair_dist[(pair_dist != positives)].view(b_size, -1)
        except RuntimeError as _:
            print(pair_dist.shape)
            print(positives.shape)
            print(negatives.shape)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(logits.shape[0]).to(self.device).long()
        # print(logits.shape, labels.shape)
        loss = self.criterion(logits, labels)
        return loss / (logits.shape[0])

    def random_crop(self, data1, data3=None):
        """Apply random cropping.

        Args:
            data1 (torch.Tensor): Batch of temporal views.
            data3 (torch.Tensor, optional): Batch of negative views. Defaults to None.

        Returns:
            (torch.Tensor): Triplet of temporal views.
        """
        crop_size = np.random.randint(self.crop_size_min, self.crop_size_max)
        #crop_size = self.crop_size_max

        max_start_index = data1.size(1) - crop_size + 1
        start_index = np.random.randint(0, max_start_index)


        crop_data_1 = data1[:, start_index : start_index + crop_size, :]

        #start_index += crop_size
        # crop_size = np.random.randint(self.crop_size_min, self.crop_size_max)
        # max_start_index = data1.size(1) - crop_size + 1
        start_index = np.random.randint(0, max_start_index)

        crop_data_2 = data1[:, start_index : start_index + crop_size, :]

        # mask_1 = self.random_mask(data1)
        # data1[~mask_1] = 0

        # mask_2 = self.random_mask(data2)
        # data2[~mask_2] = 0

        if data3 is not None:
            crop_data_3 = []
            for data in data3:
                # crop_size = np.random.randint(self.crop_size_min, self.crop_size_max)
                # max_start_index = data.size(1) - crop_size + 1
                start_index = np.random.randint(0, max_start_index)
                crop = data[:, start_index : start_index + crop_size, :]

                # mask = self.random_mask(data)
                # data[~mask] = 0

                crop_data_3.append(crop)
                # crop_data_3.append(data)
            #crop_data_3 = data3[:, start_index : start_index + crop_size, :]
        else:
            crop_data_3 = None
        return crop_data_1, crop_data_2, crop_data_3
        #return data1, data2, crop_data_3

    def temporal_triplet_loss(self, crop_z1, crop_z2, crop_z3=None, update=False):
        """Compute triplet loss using triplet of views.

        Args:
            crop_z1 (torch.Tensor): Positive crop views.
            crop_z2 (torch.Tensor): Positive crop views.
            crop_z3 (torch.Tensor, optional): Negative crop views. Defaults to None.
            update (bool, optional): Update margin. Defaults to False.

        Returns:
            torch.Tensor: Loss tensor.
        """
        # Compute the temporal loss using soft-DTW-triplet loss
        if self.use_dtw:
            loss = self.sim_loss(crop_z1, crop_z2) # positive distance

            if crop_z3 is not None:
                loss_neg_list = []
                for crop_z3_i in crop_z3:
                    loss_neg_list.append(self.sim_loss(crop_z1, crop_z3_i))
                #print(loss_neg_list[0].shape)
                loss_neg = torch.mean(torch.stack(loss_neg_list), dim=0)
                #print(loss_neg.shape)
                #loss_neg = self.sim_loss(crop_z1, crop_z3)

                # Init margin
                if self.margin is None:
                    self.margin = self.max_margin

                # pos_neg_list = []
                # for crop_z3_i in crop_z3:
                #     pos_neg_list.append(self.sim_loss(crop_z2, crop_z3_i))
                # dist_pos_neg = torch.mean(torch.stack(pos_neg_list), dim=0)

                if update:
                    # update margin

                    dist_pos_neg = self.sim_loss(crop_z2, crop_z3)
                    #print(dist_pos_neg)
                    new_margin = torch.clamp(self.max_margin - torch.mean(dist_pos_neg),
                                            min=self.min_margin)
                    #print(new_margin.item())
                    self._update_margin(new_margin.item())

                #dist = loss - .5*(loss_neg + dist_pos_neg) + self.margin
                #dist = torch.max(loss) - torch.min(loss_neg) + self.margin
                dist = loss - loss_neg + self.margin
                #print(dist.shape, loss.shape)
                loss = torch.clamp(dist, min=0.0) # triplet distance

                # Soft triplet loss
                #distance = loss - loss_neg
                #print(loss.shape)

            loss = torch.mean(loss)
            #print(loss)
            return loss
        else:
            loss = self.sim_loss(crop_z1, crop_z2, crop_z3)

    def forward(self,
                z1_batch,
                z2_batch,
                z3_batch=None,
                update=False,
                crop=True,
                cluster=False):
        """Forward pass.

        Args:
            z1_batch (torch.Tensor): Batch of positive views.
            z2_batch (torch.Tensor): Batch of 2nd positive views.
            z3_batch (torch.Tensor, optional): Batch of negative views. Defaults to None.
            update (bool, optional): Update margin. Defaults to False.
            crop (bool, optional): Apply random cropping. Defaults to True.
            cluster (bool, optional): Apply temporal clustering. Defaults to False.

        Returns:
            torch.Tensor: Loss tensor.
        """
        if self.use_dtw:
            # Perform random crop on the windows
            #print(z1_win.shape)
            #crop_z1, crop_z2, crop_z3 = self.random_crop(z1_batch, z2_batch, z3_batch)
            #data = self.random_crop(z1_batch, z2_batch, z3_batch)

            if cluster:
                # Compute temporal contrastive clustering
                loss = self.temporal_contrastive_clustering(z1_batch, z2_batch, z3_batch)

            else:
                # pos_batch = torch.cat([z1_batch, z2_batch], dim=0)
                # # Randomly select K window in the negative batchs
                # indices = torch.randint(0, pos_batch.shape[0], (z1_batch.shape[0],))
                # indices = indices.to(z1_batch.device)
                # pos_batch = torch.index_select(pos_batch, dim=0, index=indices)
                # # Compute the temporal loss using soft-DTW-triplet loss

                # loss = self._triplet_loss(pos_batch, z3_batch[0], K=self.K)
                if crop:
                    crop_z1, crop_z2, crop_z3 = self.random_crop(z1_batch, z3_batch)
                    # crop_z1, crop_z2 = self.crop(z1_batch)
                    # crop_z1_neg, crop_z2_neg = self.crop(z3_batch[0])
                    #loss1 = self.temporal_triplet_loss(crop_z1, crop_z2, [crop_z1_neg], update)
                    loss1 = self.temporal_triplet_loss(crop_z1, crop_z2, crop_z3, update)
                    crop_z1, crop_z2, crop_z3 = self.random_crop(z2_batch, z3_batch)
                    #crop_z1, crop_z2 = self.crop(z2_batch)
                    #loss2 = self.temporal_triplet_loss(crop_z1, crop_z2, [crop_z2_neg], update)
                    loss2 = self.temporal_triplet_loss(crop_z1, crop_z2, crop_z3, update)
                    loss = (loss1 + loss2) / 2
                else:
                    loss = self.temporal_triplet_loss(z1_batch, z2_batch, z3_batch)
                #loss = loss1
                #loss = self.temporal_triplet_loss(z1_batch, z2_batch, z3_batch, update=update)



            #loss = (loss1 + loss2) / 2


            # Compute the temporal loss using soft_DTW NT-Xent loss
            #print(crop_z1.shape, crop_z2.shape, crop_z3[0].shape)
            #loss = self.temporal_nt_xent_loss(crop_z1, crop_z2, crop_z3)
            #loss = self.temporal_nt_xent_loss_1(z1_batch, z2_batch, z3_batch)

        else:
            # Loop in each sample of the batch and compute timestep-wise loss
            loss = 0
            #print(z1_batch.shape)
            for i in range(z1_batch.size(0)):
                if z3_batch is not None:
                    z3_batch_i = z3_batch[i]
                else:
                    z3_batch_i = None
                loss += self.temporal_loss(z1_batch[i],
                                            z2_batch[i],
                                            z3_batch_i,
                                            update=update)
            loss /= z1_batch.size(0)
        return loss
