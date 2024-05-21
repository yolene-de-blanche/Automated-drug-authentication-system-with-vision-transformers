from PIL import Image
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

# Hyperparameters & Architecture details
NUM_CLASSES = 10
IMAGE_SIZE = 72
BATCH_SIZE = 32
EPOCHS = 100
PATCH_SIZE = 6
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
PROJECTION_DIM = 64
NUM_HEADS = 4
TRANSFORMER_UNITS = [PROJECTION_DIM * 2, PROJECTION_DIM]
MLP_HEAD_UNITS = [2048, 1024]
LEARNING_RATE = 0.00001
LAYER_NORM_EPS = 1e-6
TRANSFORMER_LAYERS = 8

# Define Multilayer Perceptron (MLP)
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.fc_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)

        for units in hidden_units:
            self.fc_layers.append(nn.Linear(input_dim, units))
            input_dim = units
        self.output_layer = nn.Linear(hidden_units[-1], output_dim)

    def forward(self, x):
        for layer in self.fc_layers:
            x = F.gelu(layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        return x

# Patch Creation Layer
class Patches(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size


    def forward(self, images):
        batch_size, width, height, channels = images.shape
        images = images.permute(0, 3, 1, 2)  # Change data shape to [batch, channels, height, width]

        # Use unfold to extract patches
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        #print(patches.shape)  # Shape will be [batch, channels, 12, 12, 6, 6]

        # Reshape to flatten the patches
        patches = patches.contiguous().view(batch_size, channels, -1, self.patch_size*self.patch_size)
        patches = patches.permute(0, 2, 1, 3)  # Reordering to [batch, num_patches, channels, patch_size*patch_size]
        patches = patches.reshape(batch_size, -1, self.patch_size*self.patch_size*channels)
        return patches

# Patch Encoding Layer
class PatchEncoder(nn.Module):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.projection = nn.Linear(108,projection_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches, projection_dim))

    def forward(self, patch):
        encoded = self.projection(patch) + self.position_embedding
        return encoded

# Multi-Head Self-Attention with Learnable Scaling
class MultiHeadAttentionLSA(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.scale = math.sqrt(self.head_dim)

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))

        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)

        x = torch.matmul(attention, V)

        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.embed_dim)

        return x

# Vision Transformer Model
class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        #self.data_augmentation = train_transform
        self.patches = Patches(PATCH_SIZE)
        self.encoder = PatchEncoder(NUM_PATCHES, PROJECTION_DIM)
        self.transformers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=PROJECTION_DIM, nhead=NUM_HEADS)
            for _ in range(TRANSFORMER_LAYERS)
        ])
        self.layer_norm = nn.LayerNorm(PROJECTION_DIM)
        self.mlp_head = MLP(PROJECTION_DIM * NUM_PATCHES, MLP_HEAD_UNITS, NUM_CLASSES)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        #x = self.data_augmentation(x)
        patches = self.patches(x)
        encoded_patches = self.encoder(patches)

        for transformer in self.transformers:
            encoded_patches = transformer(encoded_patches)
            
        representation = self.layer_norm(encoded_patches)
        representation = representation.flatten(1)
        logits = self.mlp_head(representation)
        pred = self.softmax(logits)
        return pred
