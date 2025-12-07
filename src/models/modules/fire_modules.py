import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
from torchvision.models.resnet import resnet18

np.seterr(divide='ignore', invalid='ignore')


class SimpleLSTM(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        # lstm part
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        if hparams['clc'] == 'vec':
            input_dim += 10
        hidden_size = hparams['hidden_size']
        lstm_layers = hparams['lstm_layers']
        dropout = hparams['dropout']
        self.ln1 = torch.nn.LayerNorm(input_dim)
        self.lstm = torch.nn.LSTM(input_dim, hidden_size, num_layers=lstm_layers, batch_first=True)
        # fully-connected part
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size)
        self.drop1 = torch.nn.Dropout(dropout)

        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.drop2 = torch.nn.Dropout(dropout)

        self.fc3 = torch.nn.Linear(hidden_size // 2, 2)

        self.fc_nn = torch.nn.Sequential(
            self.fc1,
            self.relu,
            self.drop1,
            self.fc2,
            self.relu,
            self.drop2,
            self.fc3
        )

    def forward(self, x: torch.Tensor):
        x = self.ln1(x)
        lstm_out, _ = self.lstm(x)
        x = self.fc_nn(lstm_out[:, -1, :])
        return torch.nn.functional.log_softmax(x, dim=1)