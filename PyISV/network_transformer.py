import math

import torch
import torch.nn.functional as F
from torch import nn

#### Compact 1D transformer autoencoder for comparison with convolutional models ####
#
# Same high-level API as FlexibleAutoencoder (embed_dim, flat_dim, input_length).
# Each distance bin is a token. flat_dim is set via AdaptiveAvgPool1d, independent
# of input_length, so you can sweep flat_dim in scan scripts the same way.


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerAutoencoder(nn.Module):
    """Small transformer autoencoder for 1D descriptors (e.g. RDF curves)."""

    def __init__(
        self,
        embed_dim,
        flat_dim,
        input_length=200,
        input_channels=1,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.embed_dim = embed_dim
        self.flat_dim = flat_dim
        self.input_length = input_length
        self.input_channels = input_channels
        self.d_model = d_model

        self.input_proj = nn.Linear(input_channels, d_model)
        self.pos_encoder = PositionalEncoding(
            d_model, max_len=max(input_length, flat_dim) + 8, dropout=dropout
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder_transformer = nn.TransformerEncoder(
            enc_layer, num_layers=num_encoder_layers
        )

        self.spatial_pool = nn.AdaptiveAvgPool1d(flat_dim)

        self.embed_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * flat_dim, embed_dim),
        )
        self.decode_linear = nn.Sequential(
            nn.Linear(embed_dim, d_model * flat_dim),
            nn.ReLU(),
        )

        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder_transformer = nn.TransformerEncoder(
            dec_layer, num_layers=num_decoder_layers
        )
        self.output_proj = nn.Linear(d_model, input_channels)

    def _encode_tokens(self, x):
        # (B, C, L) -> (B, L, d_model)
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        return self.encoder_transformer(x)

    def _encode_map(self, x):
        tokens = self._encode_tokens(x)
        maps = tokens.permute(0, 2, 1)
        return self.spatial_pool(maps)

    def _decode_map(self, z):
        # z: (B, d_model, flat_dim)
        seq = z.permute(0, 2, 1)
        seq = self.pos_encoder(seq)
        seq = self.decoder_transformer(seq)
        seq = seq.permute(0, 2, 1)
        if seq.shape[-1] != self.input_length:
            seq = F.interpolate(
                seq,
                size=self.input_length,
                mode="linear",
                align_corners=False,
            )
        seq = seq.permute(0, 2, 1)
        return self.output_proj(seq).permute(0, 2, 1)

    def forward(self, x):
        h = self._encode_map(x)
        embedding = self.embed_linear(h)
        z = self.decode_linear(embedding)
        z = z.view(z.shape[0], self.d_model, self.flat_dim)
        z = self._decode_map(z)
        return z, embedding

    def encode(self, x):
        return self.embed_linear(self._encode_map(x))

    def decode(self, embedding):
        z = self.decode_linear(embedding)
        z = z.view(z.shape[0], self.d_model, self.flat_dim)
        return self._decode_map(z)


def count_parameters(model):
    """Return trainable parameter count (useful when comparing architectures)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
