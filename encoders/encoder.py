"""
Copyright (c) 2024 Orange - All rights reserved

Author:  Joël Roman Ky
This code is distributed under the terms and conditions
of the MIT License (https://opensource.org/licenses/MIT)
"""

import torch.nn as nn

from encoders.tcn_encoder import TemporalConvNet
from encoders.ts2vec import TSEncoder

class TimeSeriesTCN(nn.Module):
    """Encoder module.
    """
    def __init__(self, input_size, proj_size, win_size, num_channels, kernel_size,
                dropout, encoder_type='lstm', num_layers=1):
        """
        Args:
            input_size (int): Input size.
            proj_size (int): Projection size.
            win_size (int): Window size.
            num_channels (list): List of channels for TCN.
            kernel_size (int): Kernel size.
            dropout (float): Dropout value.
            encoder_type (str, optional): Type of encoder. Defaults to 'lstm'.
            num_layers (int, optional): LSTM number of layers. Defaults to 1.

        Raises:
            ValueError: Encoder type not implemented.
        """
        super(TimeSeriesTCN, self).__init__()
        self.encoder_type = encoder_type
        if self.encoder_type == 'lstm':
            self.base = nn.LSTM(input_size,
                                hidden_size=num_channels[-1],
                                num_layers=num_layers)
        elif self.encoder_type == 'tcn':
            self.base = TemporalConvNet(input_size, num_channels,
                                        kernel_size=kernel_size,
                                        dropout=dropout)
        elif self.encoder_type == 'ts2vec':
            self.base = TSEncoder(input_dims=input_size, output_dims=num_channels[-1])
        else:
            raise ValueError('Encoder type not implemented')

        feat_size = num_channels[-1]*win_size
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


    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Tensor batch.

        Returns:
            (torch.Tensor*): Latent vector, projection vector.
        """
        if self.encoder_type == 'lstm':
            feat, _ = self.base(x)
        elif self.encoder_type == 'tcn':
            out = x.permute(0, 2, 1)
            z_i = self.base(x)
            feat = z_i.permute(0, 2, 1)
        elif self.encoder_type == 'ts2vec':
            feat = self.base(x)

        out = feat.reshape(feat.size(0), -1)
        v_i = self.projection(out)
        return v_i, feat
