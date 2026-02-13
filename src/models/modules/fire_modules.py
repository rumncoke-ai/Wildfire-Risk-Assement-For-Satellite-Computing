import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
from torchvision.models import mobilenet_v2
from src.models.modules.convlstm import ConvLSTM


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
            nn.ReLU(),
            nn.MaxPool2d(2),  # 25×25 → 12×12
            nn.Dropout(dropout)
        )
        
        # Calculate flattened size
        # Assuming input is 25×25, after pool: 12×12
        fc_input_size = 12 * 12 * 16
        
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.conv_block(x)
        x = torch.flatten(x, 1)
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


class MobileNetV2CNN(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        
        # Get input channels (static + dynamic features)
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        
        # Load pretrained MobileNetV2
        self.mobilenet = mobilenet_v2(pretrained=True)
        
        # Modify the first conv layer to accept our input channels
        # Original MobileNetV2 expects 3 channels (RGB), we need to adapt it
        if input_dim != 3:
            # Replace first convolutional layer to accept input_dim channels
            original_conv = self.mobilenet.features[0][0]
            self.mobilenet.features[0][0] = nn.Conv2d(
                input_dim, 
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            
            # Initialize the new layer weights properly
            nn.init.kaiming_normal_(self.mobilenet.features[0][0].weight, mode='fan_out')
            if self.mobilenet.features[0][0].bias is not None:
                nn.init.zeros_(self.mobilenet.features[0][0].bias)
        
        # Get the number of features from the last layer
        num_features = self.mobilenet.classifier[1].in_features
        
        # Replace classifier with our own (keep dropout if exists)
        dropout = hparams.get('dropout', 0.2)
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 2)
        )
        
        # Alternatively, you can create a custom head
        # self.mobilenet.classifier = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(num_features, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(dropout),
        #     nn.Linear(256, 2)
        # )
    
    def forward(self, x: torch.Tensor):
        x = self.mobilenet(x)
        return torch.nn.functional.log_softmax(x, dim=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()


        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)


        div_term = torch.exp(
        torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )


        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)


        pe = pe.unsqueeze(0) # (1, T, D)
        self.register_buffer("pe", pe)


    def forward(self, x: torch.Tensor):
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T]

class TinyTemporalTransformer(nn.Module):
    """
    Designed for small scientific time‑series:


    Input: (B, T, C)
    Output: log-probabilities for binary classification


    Small parameter count → suitable for edge/satellite experiments.
    """


    def __init__(self, hparams: dict):
        super().__init__()


        input_dim = len(hparams["dynamic_features"]) + len(hparams["static_features"])
        if hparams.get("clc") == "vec":
            input_dim += 10


        d_model = hparams.get("hidden_size", 64)
        n_heads = hparams.get("n_heads", 4)
        n_layers = hparams.get("n_layers", 2)
        dropout = hparams.get("dropout", 0.1)


        # Feature embedding
        self.input_proj = nn.Linear(input_dim, d_model)
        self.ln = nn.LayerNorm(d_model)


        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)


        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=d_model * 4,
        dropout=dropout,
        batch_first=True,
        activation="gelu",
        norm_first=True,
        )


        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)


        # Classification head (mean pooling)
        self.classifier = nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Dropout(dropout),
        nn.Linear(d_model, 2),
        )


    def forward(self, x: torch.Tensor):
        # x: (B, T, C)
        x = self.input_proj(x)
        x = self.ln(x)
        x = self.pos_encoder(x)


        x = self.transformer(x) # (B, T, D)


        # Mean pooling over time
        x = x.mean(dim=1)


        x = self.classifier(x)
        return F.log_softmax(x, dim=1)