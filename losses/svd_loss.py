"""
Copyright (c) 2024 Orange - All rights reserved

Author:  Joël Roman Ky
This code is distributed under the terms and conditions
of the MIT License (https://opensource.org/licenses/MIT)
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class SVDLoss(torch.nn.Module):
    """Deep-SVDD loss.
    """
    def __init__(self, radius, device, center=None, nu_val=.1, objective='soft'):
        """
        Args:
            radius (float): Hypersphere radius.
            device (torch.device): Device.
            center (torch.Tensor, optional): Hypersphere center.. Defaults to None.
            nu_val (float, optional): Anomaly rate. Defaults to .1.
            objective (str, optional): Soft or Hard hypersphere. Defaults to 'soft'.
        """
        super(SVDLoss, self).__init__()
        self.device = device
        if isinstance(radius, float):
            self.radius = torch.tensor(radius).to(self.device)
        else:
            self.radius = radius.to(self.device)
        if center is not None:
            self.center = center.to(self.device)
        else:
            self.center = center
        self.objective = objective
        self.nu_val = nu_val
        self.cosine_fn = nn.CosineSimilarity(dim=-1)

    def forward(self, z_embed, update=False, test=False, cosine=False):
        """Forward pass.

        Args:
            z_embed (torch.Tensor): Batch vectors.
            update (bool, optional): Update radius. Defaults to False.
            test (bool, optional): Inference. Defaults to False.
            cosine (bool, optional): Apply cosine distance. Defaults to False.

        Returns:
            torch.Tensor: Score/Loss value.
        """
        dist = torch.sum((z_embed - self.center)**2, dim=-1)
        #dist = torch.linalg.norm(z_embed - self.center, ord=2, dim=-1)

        if self.objective=='soft':
            scores = dist - self.radius **2
            loss = self.radius**2 + (1/self.nu_val) \
                * torch.mean(torch.max(torch.zeros_like(scores), scores))

            if update:
                self._update_radius(dist)
        else:
            scores = dist
            loss = torch.mean(dist)

        if test:
            if cosine:
                score = self.cosine_fn(F.normalize(z_embed, dim=-1),
                                        F.normalize(self.center.unsqueeze(0), dim=-1))
                scores = 1 - score
            return scores
        else:
            return loss

    def init_center(self, model, train_loader, win_size, embedding_size, eps=.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data.
        https://github.com/lukasruff/Deep-SVDD-PyTorch/blob/master/src/optim/deepSVDD_trainer.py

        Args:
            model (torch.nn): Torch model.
            train_loader (DataLoader): Train dataloader
            win_size (int): Window size.
            embedding_size (int): Embedding size.
            eps (float, optional): Epsilon value. Defaults to 0.1.

        Returns:
            torch.Tensor: Hyperspace center.
        """
        n_samples = 0
        center = torch.zeros(win_size, embedding_size).to(self.device)


        model.eval()
        model = model.to(self.device)
        with torch.no_grad():
            for batch in train_loader:
                # get the inputs of the batch
                positive_1 = batch[0]
                positive_1 = positive_1.to(self.device)
                outs = model(positive_1)
                pos_feat_1 = outs[0]

                n_samples += pos_feat_1.shape[0]
                center += torch.sum(pos_feat_1, dim=0)


        center /= n_samples

        # If c_i is too close to 0, set to +-eps.
        # Reason: a zero unit can be trivially matched with zero weights.
        center[(abs(center) < eps) & (center < 0)] = -eps
        center[(abs(center) < eps) & (center > 0)] = eps
        self.center = center
        return center

    def _update_radius(self, dist: torch.Tensor):
        """Get hypersphere radius.

        Args:
            dist (torch.Tensor): _description_
            nu_val (float): Outliers hyperparameters.

        Returns:
            np.array: Radius array.
        """
        #self.radius = np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1-self.nu_val)
        self.radius = torch.tensor(np.quantile(np.sqrt(dist.clone().data.cpu().numpy()),
                                                1-self.nu_val)).to(self.device)
