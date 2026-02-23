import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
from torchvision.models import mobilenet_v2, resnet18
from src.models.modules.convlstm import ConvLSTM
import math


np.seterr(divide='ignore', invalid='ignore')


class SimpleConvLSTM(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        if hparams['clc'] == 'vec':
            input_dim += 10
        hidden_size = hparams['hidden_size']
        lstm_layers = hparams['lstm_layers']
        dropout = hparams['dropout']
        kernel_size = 3
        self.ln1 = torch.nn.LayerNorm(input_dim)
        # clstm part
        self.convlstm = ConvLSTM(input_dim,
                                 hidden_size,
                                 (kernel_size, kernel_size),
                                 lstm_layers,
                                 True,
                                 True,
                                 False, dilation=1)

        self.conv1 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(kernel_size, kernel_size), stride=(1, 1),
                               padding=(1, 1))
        # fully-connected part
        self.fc1 = nn.Linear((25 // 2) * (25 // 2) * hidden_size, 2 * hidden_size)
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(2 * hidden_size, hidden_size)
        self.drop2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor):
        # (b x t x c x h x w) -> (b x t x h x w x c) -> (b x t x c x h x w)
        x = self.ln1(x.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        _, last_states = self.convlstm(x)
        x = last_states[0][0]
        # cnn
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # fully-connected
        x = torch.flatten(x, 1)
        x = F.relu(self.drop1(self.fc1(x)))
        x = F.relu(self.drop2(self.fc2(x)))
        x = self.fc3(x)
        return torch.nn.functional.log_softmax(x, dim=1)


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

class SimpleCNN(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        # if hparams['clc'] == 'vec':
        #     input_dim += 10
        
        hidden_size = hparams['hidden_size']
        dropout = hparams['dropout']
        
        # Match spatial dataset architecture
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_dim, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 25×25 → 12×12
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # # Calculate flattened size
        # # Assuming input is 25×25, after pool: 12×12
        # fc_input_size = 12 * 12 * hidden_size
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

        self.flatten = nn.Flatten()
    
    def forward(self, x: torch.Tensor):
        x = self.conv_block(x)
        x = self.flatten(x)
        x = self.fc(x)
        return torch.nn.functional.log_softmax(x, dim=1)


class Simple1DCNN(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        if hparams['clc'] == 'vec':
            input_dim += 10
        
        hidden_size = hparams['hidden_size']
        dropout = hparams['dropout']
        
        # Match spatial dataset architecture
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_dim, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

       
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
    
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)
        x = self.conv_block(x)
        x = x.mean(dim=2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=1)


class ResNet18CNN(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        
        # Get input channels (static + dynamic features)
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        dropout = hparams.get('dropout', 0.2)

        # Load ResNet18 (recommended: pretrained=False for non-image data)
        self.resnet = resnet18(weights=None)

        # 🔧 Modify first conv layer for multi-channel + small inputs
        self.resnet.conv1 = nn.Conv2d(
            input_dim,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # ❌ Remove maxpool (important for 25×25 inputs)
        self.resnet.maxpool = nn.Identity()

        # Replace classifier
        num_features = self.resnet.fc.in_features

        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, x: torch.Tensor):
        x = self.resnet(x)
        return torch.nn.functional.log_softmax(x, dim=1)

class TinyViT(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        if hparams['clc'] == 'vec':
            input_dim += 10

        image_size = 25
        patch_size = 5
        num_patches = (image_size // patch_size) ** 2
        embed_dim = hparams['hidden_size']
        num_heads = 4
        num_layers = 2
        dropout = hparams['dropout']

        self.patch_embed = nn.Conv2d(
            input_dim,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 2)

    def forward(self, x):
        x = self.patch_embed(x)  # (B, embed_dim, 5, 5)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embedding
        x = self.transformer(x)

        cls_output = x[:, 0]
        cls_output = self.norm(cls_output)

        logits = self.head(cls_output)

        return F.log_softmax(logits, dim=1)