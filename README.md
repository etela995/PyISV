# PyISV

Python library for training convolutional autoencoder-style networks on structural descriptors (especially RDFs from atomic configurations), as used in:

- [arXiv:2407.17924](https://doi.org/10.48550/arXiv.2407.17924)
- [ACS Nano (2023)](https://doi.org/10.1021/acsnano.3c05653)

## Installation

Clone the repository and install in your environment:

```bash
pip install .
# or, for development:
pip install -e .
```

Requires Python 3.8+ and PyTorch.

## Quick start

Typical workflow (edit paths and parameters in each script before running):

1. **Compute descriptors** from an ASE-readable trajectory (e.g. XYZ):

   ```bash
   cd Scripts/1D
   # Edit file.xyz, paths, and options in compute_single_rdf.py
   python compute_single_rdf.py
   ```

   Output: `rdfs.npy` with shape `(n_frames, n_bins)`.

2. **Train** the autoencoder on your `.npy` data:

   ```bash
   # Edit input_path, embed_dim, flat_dim, etc. in model_training_script.py
   python model_training_script.py
   ```

   Outputs: `best_model.pth`, scaler `.npy` files, `train_log.txt`, `train_stats.txt`.

3. **Evaluate** embeddings and reconstructions:

   ```bash
   python model_evaluation_script.py
   ```

   Output: `embed.npy` (and reconstructions if enabled).

For **binary** systems with three partial RDFs, use `compute_triple_rdf.py` and the 2D scripts under `Scripts/2D/`.

## Repository layout

| Path | Description |
|------|-------------|
| `PyISV/features_calc_utils.py` | KDE-based distance histograms (RDF-style descriptors) from ASE `Atoms` |
| `PyISV/network.py` | Default **1D** autoencoder: 200 bins per channel, `flat_dim=1`, multi-channel support |
| `PyISV/network_flex.py` | **1D** autoencoder with **user-set `flat_dim`** at fixed `input_length` (adaptive pool + resize) |
| `PyISV/network_transformer.py` | **Compact transformer** 1D autoencoder (same API as `FlexibleAutoencoder` for comparisons) |
| `PyISV/network_arxiv.py` | **1D** architecture from the arXiv paper: 340 bins, `flat_dim=21` |
| `PyISV/network_2D.py` | **2D** autoencoder for matrix inputs |
| `PyISV/train_utils.py` | Dataset normalization, losses, checkpointing, early stopping |
| `Scripts/1D/` | RDF computation and 1D training/evaluation examples |
| `Scripts/2D/` | 2D training/evaluation examples |

Other `network_*.py` modules are alternate architectures for specific input shapes; match their docstrings/comments when changing input size.

## Choosing `flat_dim`

`flat_dim` must match the **spatial width** of the tensor leaving the encoder (before the bottleneck linear layer):

- **1D** (`network.py`): after seven max-pool steps on length 200, the encoder width is **1** → use `flat_dim=1`.
- **1D** (`network_arxiv.py`): tuned for length **340** → use `flat_dim=21`.
- **2D** (`network_2D.py`): `flat_dim` is the product `H × W` of the encoder feature map (must be a perfect square for the built-in reshape).

If you change input length or architecture, run a dry forward pass and read the encoder output shape. You can also use:

```python
from PyISV.train_utils import infer_flat_dim
import torch

model = Autoencoder(embed_dim=2, flat_dim=1)  # trial value
sample = torch.zeros(1, 1, 200)  # (batch, channels, length)
print(infer_flat_dim(model, sample))  # -> 1 for default network.py + length 200
```

The training script prints a `torchsummary` table; the commented line in `network.py` forward (`Flat dim should be:`) refers to the same quantity.

If `flat_dim` is wrong, training will fail with a linear-layer size mismatch—set it to the value from `infer_flat_dim`.

### Flexible `flat_dim` (fixed input length)

Use `FlexibleAutoencoder` when you want to try different bottleneck widths **without** changing RDF length or retuning decoder padding:

```python
from PyISV import FlexibleAutoencoder
import torch

model = FlexibleAutoencoder(
    embed_dim=2,
    flat_dim=8,        # try 1, 5, 8, 21, ... at the same input_length
    input_length=200,  # must match your descriptor length
    input_channels=1,
)
x = torch.randn(4, 1, 200)
recon, z = model(x)
assert recon.shape == x.shape
```

In `Scripts/1D/model_training_script.py`, set `use_flexible_autoencoder = True` and `input_length` to your bin count.

### Transformer vs convolutional

For **fixed-length 1D RDFs** (~200 bins), convolutional models are often a strong baseline: they encode local peak/shape patterns with few parameters and train quickly. **Transformers** attend over all bins globally, which can help when long-range patterns across the distance axis matter, but on short sequences they may **not outperform** a tuned conv net unless you have enough data or need global mixing at the bottleneck.

`TransformerAutoencoder` is intentionally **small** (`d_model=64`, 2 encoder + 2 decoder layers) so you can compare fairly:

```python
from PyISV import FlexibleAutoencoder, TransformerAutoencoder, count_parameters

kw = dict(embed_dim=2, flat_dim=8, input_length=200, input_channels=1)
conv = FlexibleAutoencoder(**kw)
tr = TransformerAutoencoder(**kw)
print(count_parameters(conv), count_parameters(tr))
```

Swap the model class in `scan_flat_dim_script.py` (same `flat_dims_to_scan` and data pipeline).

To **compare several `flat_dim` values** on the same `.npy` file (with optional resampling to 200 bins), use `Scripts/1D/scan_flat_dim_script.py`. After training, run `Scripts/1D/evaluate_flat_dim_scan_script.py` to print `torchsummary` for each checkpoint and save embedding vectors.

## Input shapes

- Default **1D** model: each channel is a vector of **200** values (e.g. 200 RDF bins). Multi-channel inputs use shape `(N, C, 200)`.
- **arxiv** 1D model: **340** values per channel, `flat_dim=21`.
- Descriptors need not come from this repo; any array with the correct shape works (padding or interpolation is fine).

## Remarks

The networks are general convolutional autoencoders: descriptors can be computed elsewhere (e.g. scikit-learn KDE) as long as tensor shapes match the chosen architecture. Adjust decoder padding and `flat_dim` when input dimensions change.
