import torch
import torch.nn.functional as F
from torch import nn

#### Flexible 1D autoencoder: flat_dim is independent of input_length ####
#
# Encoder convolutions + AdaptiveAvgPool1d(flat_dim) set the bottleneck width.
# Decoder upsamples from flat_dim and interpolates to input_length at the end,
# so you can change flat_dim without retuning paddings or changing RDF bin count.


class FlexibleAutoencoder(nn.Module):

    def __init__(
        self,
        embed_dim,
        flat_dim,
        input_length=200,
        input_channels=1,
        kernel_size=5,
        num_final_channels=128,
        encoder_pools=4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.flat_dim = flat_dim
        self.input_length = input_length
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.num_final_channels = num_final_channels

        self.encoder = self._build_encoder(encoder_pools)
        self.spatial_pool = nn.AdaptiveAvgPool1d(flat_dim)

        self.embed_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_final_channels * flat_dim, embed_dim),
        )
        self.decode_linear = nn.Sequential(
            nn.Linear(embed_dim, num_final_channels * flat_dim),
            nn.ReLU(),
        )
        self.decoder = self._build_decoder()

    def _build_encoder(self, num_pools):
        channels = [8, 16, 32, 64, 64, 128, 128]
        layers = []
        in_ch = self.input_channels
        for i, out_ch in enumerate(channels):
            layers.append(
                nn.Conv1d(
                    in_ch,
                    out_ch,
                    kernel_size=self.kernel_size,
                    padding="same",
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(out_ch))
            if i < num_pools:
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_ch = out_ch
        return nn.Sequential(*layers)

    def _build_decoder(self):
        specs = [
            (128, 128),
            (128, 64),
            (64, 64),
            (64, 32),
            (32, 16),
            (16, 8),
        ]
        layers = []
        for in_ch, out_ch in specs:
            layers += [
                nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
                nn.Conv1d(
                    in_ch,
                    out_ch,
                    kernel_size=self.kernel_size,
                    padding="same",
                ),
                nn.ReLU(),
                nn.BatchNorm1d(out_ch),
            ]
        layers.append(
            nn.Conv1d(
                8,
                self.input_channels,
                kernel_size=self.kernel_size,
                padding="same",
            )
        )
        return nn.Sequential(*layers)

    def _encode_map(self, x):
        x = self.encoder(x)
        return self.spatial_pool(x)

    def _decode_map(self, z):
        z = self.decoder(z)
        if z.shape[-1] != self.input_length:
            z = F.interpolate(
                z,
                size=self.input_length,
                mode="linear",
                align_corners=False,
            )
        return z

    def forward(self, x):
        h = self._encode_map(x)
        embedding = self.embed_linear(h)
        z = self.decode_linear(embedding)
        z = z.view(z.shape[0], self.num_final_channels, self.flat_dim)
        z = self._decode_map(z)
        return z, embedding

    def encode(self, x):
        return self.embed_linear(self._encode_map(x))

    def decode(self, embedding):
        z = self.decode_linear(embedding)
        z = z.view(z.shape[0], self.num_final_channels, self.flat_dim)
        return self._decode_map(z)
