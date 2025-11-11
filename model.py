import sys
import sofar
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50, ResNet50_Weights

class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()

        # Load a pretrained ResNet50 model
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Modify the first convolutional layer to accept 1 channel
        self.resnet = resnet
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=resnet.conv1.out_channels,
            kernel_size=resnet.conv1.kernel_size,
            stride=resnet.conv1.stride,
            padding=resnet.conv1.padding,
            bias=resnet.conv1.bias
        )

        # Initialize weights of the modified layer
        nn.init.kaiming_normal_(self.resnet.conv1.weight, mode="fan_out", nonlinearity="relu")

        self.encoder = nn.Sequential(*list(resnet.children())[:-2])


    def forward(self, x):
        batch_size, sides, views, channels, height, width = x.size()
        x = x.view(batch_size * sides * views, channels, height, width)  # Shape: [batch_size * sides * views, 1, 256, 256]

        # Forward pass through ResNet
        features = self.encoder(x)  # [batch, 2048, H', W']
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))  # [batch_size * sides * views, 2048, 1, 1]
        out = features.view(batch_size * sides * views, 2048)  # [batch_size * sides * views, 2048]

        # Aggregate features across views and sides
        out = out.view(batch_size, views * sides, 2048)  # [batch_size, views * sides, 2048]

        return out


class HRTFModel(nn.Module):
    def __init__(self):
        super(HRTFModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet_encoder = ResNetEncoder()

        # LSTM for image features
        self.lstm = nn.LSTM(input_size=2048,
                    hidden_size=1024,
                    num_layers=4,
                    bidirectional=True,
                    batch_first=True,
                    dropout=0.3)

        # LSTM for expansion of directions
        self.expansion = nn.LSTM(input_size=1,
                                hidden_size=793,
                                num_layers=2,
                                batch_first=True,
                                bidirectional=True,
                                dropout=0.3)

        # Output layer for signal prediction
        self.out = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )

    def forward(self, x):
        # ResNet features
        batch_size, sides, views = x.size(0), x.size(1), x.size(2)
        resnet_out = self.resnet_encoder(x)  # Shape: [batch_size, sides * views, 2048]

        ### THIS IS USELESS! LSTM CAN HANDLE VARIABLE LENGTH INPUTS ###
        ### FIX THIS! ###
        if sides * views != 6:
            resnet_out = F.adaptive_avg_pool1d(resnet_out.permute(0, 2, 1), 6).permute(0, 2, 1)

        o, (h, _) = self.lstm(resnet_out)
        summary = torch.cat((h[-2], h[-1]), dim=-1)
        summary = summary.unsqueeze(1)

        features = summary.view(batch_size, -1, 1)
        expanded_features, _ = self.expansion(features)
        expanded_features = expanded_features.contiguous().view(batch_size, 1586, 2048)

        # Output layer
        x = self.out(expanded_features)

        x = x.view(batch_size, 793, 2, 256)

        return x