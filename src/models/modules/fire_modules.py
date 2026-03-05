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

    def forward(self, x: torch.Tensor):
        x = self.resnet(x)
        return torch.nn.functional.log_softmax(x, dim=1)


class StaticTinyViT(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        self.input_dim = len(hparams['static_features'])
        image_size = 25
        patch_size = 5
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = hparams['hidden_size']

        self.patch_embed = nn.Conv2d(
            self.input_dim,
            self.embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.embed_dim)
        ) 

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=4, 
            dim_feedforward=self.embed_dim * 2,
            dropout=hparams['dropout'],
            activation='gelu',
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.norm = nn.LayerNorm(self.embed_dim)
        
        # Initialize positional embedding
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)  # (B, 64, 5, 5)
        x = x.flatten(2).transpose(1, 2)  # (B, 25, 64)
        
        # Add CLS token
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 26, 64)
        
        # Add positional embedding
        x = x + self.pos_embedding
        
        # Transformer
        x = self.transformer(x)
        
        # Take CLS token output
        cls_output = x[:, 0]
        cls_output = self.norm(cls_output)
        
        return cls_output  # (B, 64)

class DynamicMobileViT(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        self.input_dim = len(hparams['dynamic_features'])
        
        self.time_steps = 10
        self.embed_dim = hparams['hidden_size']

        self.patch3d = nn.Conv3d(
                in_channels=self.input_dim,
                out_channels=self.embed_dim,
                kernel_size=(5,5,5),
                stride=(5,5,5)
        )

        # Compute number of 3D patches:
        # Input: (T=10, H=25, W=25)
        # Kernel/stride = 5
        # Depth: (10-5)/5+1 = 2
        # Height: 5
        # Width: 5
        self.num_patches = 2 * 5 * 5  # 50 tokens

        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, self.embed_dim)
        )


        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=4,
                dim_feedforward=self.embed_dim * 2,
                dropout=hparams['dropout'],
                batch_first=True
            ),
            num_layers=2
        )

        self.norm = nn.LayerNorm(self.embed_dim) 

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = self.patch3d(x)  # (B, D, 2, 5, 5)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = self.norm(x.mean(dim=1))

        return x

class ResidualFusionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

        self.final_norm = nn.LayerNorm(dim)

    def forward(self, static_feat, dynamic_feat):
        # static_feat: (B, D)
        # dynamic_feat: (B, D)

        x = torch.stack([static_feat, dynamic_feat], dim=1)  # (B, 2, D)

        # ---- Residual Attention ----
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]

        # ---- Residual MLP ----
        x = x + self.mlp(self.norm2(x))

        # Pool tokens (mean pooling)
        fused = self.final_norm(x.mean(dim=1))  # (B, D)

        return fused


class LateFusionAttention(nn.Module):
    def __init__(self, dim=128):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=4,
            batch_first=True
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, static_feat, dynamic_feat):
        x = torch.stack([static_feat, dynamic_feat], dim=1)
        attn_out, _ = self.attn(x, x, x)
        out = self.norm(attn_out.mean(dim=1))
        return out

class DualBranchViT(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.static_branch = StaticTinyViT(hparams)
        self.dynamic_branch = DynamicMobileViT(hparams)

        hidden_size = hparams['hidden_size']

        # self.fusion = LateFusionAttention(dim=hidden_size)

        self.fusion = ResidualFusionBlock(
            dim=hidden_size,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=hparams['dropout']
        )

        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, static, dynamic):
        static = static.float()
        dynamic = dynamic.float()

        static_feat = self.static_branch(static)
        dynamic_feat = self.dynamic_branch(dynamic)

        fused = self.fusion(static_feat, dynamic_feat)
        logits = self.classifier(fused)

        return F.log_softmax(logits, dim=1)

# class DynamicCNNStem(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()

#         # Initial spatio-temporal feature extraction
#         self.temporal = nn.Conv3d(
#             in_channels,
#             32,
#             kernel_size=(3, 3, 3),
#             padding=(1, 1, 1)
#         )

#         # Temporal self-attention
#         self.temporal_attn = nn.MultiheadAttention(
#             embed_dim=32,
#             num_heads=4,
#             batch_first=True
#         )
        

#         self.temporal_gate = nn.Linear(32, 1)

#         # Spatial CNN refinement
#         self.spatial = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.GELU(),

#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.GELU(),
#         )

#     def forward(self, x):
#         # Input: (B, T, C, H, W)

#         B, T, C, H, W = x.shape

#         # Rearrange for Conv3D
#         x = x.permute(0, 2, 1, 3, 4)   # (B, C, T, H, W)

#         # Spatio-temporal convolution
#         x = self.temporal(x)          # (B, 32, T, H, W)

#         # Prepare for temporal attention
#         # Each spatial pixel becomes a sequence over time
#         x = x.permute(0, 3, 4, 2, 1)  # (B, H, W, T, C)

#         x = x.reshape(B * H * W, T, 32)  # (B*H*W, T, C)

#         # # Temporal self-attention
#         x, _ = self.temporal_attn(x, x, x)

#         scores = self.temporal_gate(x)        # (BHW, T, 1)
#         weights = torch.softmax(scores, dim=1)

#         x = (x * weights).sum(dim=1)

#         # # Aggregate time dimension
#         # x = x.mean(dim=1)  # (B*H*W, C)

#         # Restore spatial structure
#         x = x.reshape(B, H, W, 32).permute(0, 3, 1, 2)  # (B, 32, H, W)

#         # Spatial CNN
#         x = self.spatial(x)  # (B, 64, H, W)

#         return x

class DynamicCNNStem(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.temporal = nn.Conv3d(
            in_channels,
            32,
            kernel_size=(3,3,3),
            padding=(1,1,1)
        )

        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=4,
            batch_first=True
        )

        # Normalization for transformer stability
        self.temporal_norm = nn.LayerNorm(32)

        # Temporal importance gate
        self.temporal_gate = nn.Linear(32,1)

        # Spatial CNN
        self.spatial = nn.Sequential(
            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

    def forward(self,x):

        B,T,C,H,W = x.shape

        # Conv3D expects (B,C,T,H,W)
        x = x.permute(0,2,1,3,4)

        x = self.temporal(x)      # (B,32,T,H,W)

        # Prepare temporal sequences per pixel
        x = x.permute(0,3,4,2,1)  # (B,H,W,T,C)
        x = x.reshape(B*H*W, T, 32)

        # ---- Temporal Transformer ----

        residual = x
        attn_out,_ = self.temporal_attn(x,x,x)

        x = self.temporal_norm(residual + attn_out)

        # ---- Temporal importance weighting ----

        scores = self.temporal_gate(x)      # (BHW,T,1)
        weights = torch.softmax(scores,dim=1)

        x = (x * weights).sum(dim=1)

        # Restore spatial layout
        x = x.reshape(B,H,W,32).permute(0,3,1,2)

        # Spatial CNN refinement
        x = self.spatial(x)

        return x



class SpatialTinyTransformer(nn.Module):
    def __init__(self, embed_dim=32, num_layers=2, dropout=0.5):
        super().__init__()

        self.patch_embed = nn.Conv2d(
            64,
            embed_dim,
            kernel_size=5,
            stride=5
        )

        num_patches = (25 // 5) ** 2  # 25 tokens

        self.pos_embedding = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)              # (B, D, 5, 5)
        x = x.flatten(2).transpose(1, 2)     # (B, 25, D)

        x = x + self.pos_embedding
        x = self.transformer(x)

        x = self.norm(x)
        x = x.mean(dim=1)

        return x

class StaticCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.net(x)
        x = self.pool(x).flatten(1)
        return x

class HybridCNNViT(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        dynamic_channels = len(hparams['dynamic_features'])
        static_channels = len(hparams['static_features'])

        self.dynamic_stem = DynamicCNNStem(dynamic_channels)
        self.transformer_head = SpatialTinyTransformer(
            embed_dim=32,
            num_layers=2,
            dropout=0.5
        )

        self.static_branch = StaticCNN(static_channels)

        self.classifier = nn.Sequential(
            nn.Linear(32 + 32, 64),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )

    def forward(self, static, dynamic):
        static = static.float()
        dynamic = dynamic.float()

        dyn_feat_map = self.dynamic_stem(dynamic)
        dyn_feat = self.transformer_head(dyn_feat_map)

        stat_feat = self.static_branch(static)

        # fused = torch.cat([dyn_feat, stat_feat], dim=1)

        gamma = self.gamma(stat_feat)
        beta = self.beta(stat_feat)

        dyn_feat = gamma * dyn_feat + beta

        fused = torch.cat([dyn_feat, stat_feat], dim=1)

        logits = self.classifier(fused)

        return F.log_softmax(logits, dim=1)