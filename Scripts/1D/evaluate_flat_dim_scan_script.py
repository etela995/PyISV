"""Load all flat_dim scan checkpoints, print torchsummary, and save embeddings.

Also saves scaled-back reconstructions in the same units as the original inputs
for direct comparison. Expects models from scan_flat_dim_script.py and scaler
files in the same output_dir. Edit PARAMETERS to match the scan run.

Run from Scripts/1D:

    python evaluate_flat_dim_scan_script.py
"""
import glob
import os
import re
import time

import numpy as np
import torch
from torchsummary import summary
from tqdm import tqdm

from PyISV.network_flex import FlexibleAutoencoder
from PyISV.network_transformer import TransformerAutoencoder
from PyISV.train_utils import resample_descriptors

##########################
####### PARAMETERS #######
##########################

input_path = '/Users/tela/ubscratch/Photonics/StefanoData/merged_spectra.npy'
input_dimensionality = 1

# Must match scan_flat_dim_script.py
length_mode = "resample"
input_length = 200
embed_dim = 2
input_channels = None

models_dir = "flat_dim_scan"
ModelClass = FlexibleAutoencoder  # or TransformerAutoencoder (must match scan)
model_glob = "best_model_flat*.pth"
scaler_subval_path = None  # None -> models_dir/input_scaler_subval.npy
scaler_divval_path = None

device = "cpu"
eval_batch_size = 64

embeddings_dir = None  # None -> models_dir/embeddings
original_inputs_name = "original_inputs.npy"  # pre-normalization inputs (for comparison)
summary_file = None  # None -> models_dir/model_summaries.txt


def discover_checkpoints(directory, pattern):
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    cwd_paths = sorted(glob.glob(pattern))
    seen = set()
    unique = []
    for p in paths + cwd_paths:
        abspath = os.path.abspath(p)
        if abspath not in seen:
            seen.add(abspath)
            unique.append(abspath)
    return unique


def parse_flat_dim(path):
    match = re.search(r"flat(\d+)", os.path.basename(path))
    if not match:
        raise ValueError(f"Cannot parse flat_dim from checkpoint name: {path}")
    return int(match.group(1))


def load_and_prepare_data():
    raw = np.load(input_path)
    native_length = raw.shape[-1]

    if length_mode == "resample":
        network_length = input_length
        if native_length != network_length:
            print(
                f"Resampling descriptors: {native_length} bins -> {network_length} bins"
            )
            raw = resample_descriptors(raw, network_length)
    elif length_mode == "native":
        network_length = native_length
    else:
        raise ValueError('length_mode must be "resample" or "native"')

    if len(raw.shape) > input_dimensionality + 1:
        num_channels = raw.shape[1] if input_channels is None else input_channels
        tensor = torch.tensor(raw, dtype=torch.double)
    else:
        num_channels = 1 if input_channels is None else input_channels
        tensor = torch.tensor(raw, dtype=torch.double).unsqueeze(1)

    return tensor, network_length, num_channels


def apply_scalers(tensor, subval, divval):
    sub = torch.tensor(subval, dtype=torch.double)
    div = torch.tensor(divval, dtype=torch.double)
    dataset_size = tensor.shape[0]
    sample_shape = tensor.shape[1:]
    sub = sub.unsqueeze(0).expand(dataset_size, *sample_shape)
    div = div.unsqueeze(0).expand(dataset_size, *sample_shape)
    return tensor.sub(sub).div(div)


def scale_back(array, subval, divval):
    """Invert min-max (or gaussian) normalization used during training."""
    arr = np.asarray(array, dtype=np.float64)
    sub = np.asarray(subval, dtype=np.float64)
    div = np.asarray(divval, dtype=np.float64)
    while sub.ndim < arr.ndim:
        sub = np.expand_dims(sub, 0)
        div = np.expand_dims(div, 0)
    return arr * div + sub


def array_to_saved_inputs(arr):
    """Save as (N, L) or (N, C, L) matching the loaded .npy layout."""
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[1] == 1:
        return arr[:, 0, :]
    return arr


@torch.no_grad()
def forward_model(model, scaled_tensor, batch_size):
    model.eval()
    embeddings = []
    reconstructions = []
    n = scaled_tensor.shape[0]

    for start in tqdm(range(0, n, batch_size), desc="  batches", leave=False):
        batch = scaled_tensor[start : start + batch_size].float().to(device)
        out, emb = model(batch)
        embeddings.append(emb.cpu().numpy())
        reconstructions.append(out.cpu().numpy())

    return np.vstack(embeddings), np.concatenate(reconstructions, axis=0)


def main():
    out_dir = models_dir
    emb_dir = embeddings_dir or os.path.join(out_dir, "embeddings")
    summ_path = summary_file or os.path.join(out_dir, "model_summaries.txt")
    os.makedirs(emb_dir, exist_ok=True)

    sub_path = scaler_subval_path or os.path.join(out_dir, "input_scaler_subval.npy")
    div_path = scaler_divval_path or os.path.join(out_dir, "input_scaler_divval.npy")
    if not os.path.isfile(sub_path) or not os.path.isfile(div_path):
        raise FileNotFoundError(
            f"Scaler files not found in {out_dir}. Run scan_flat_dim_script.py first "
            "or set scaler_subval_path / scaler_divval_path."
        )

    checkpoints = discover_checkpoints(out_dir, model_glob)
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints matching {model_glob} in {out_dir} or current directory."
        )

    subval = np.load(sub_path)
    divval = np.load(div_path)

    tensor, network_length, num_channels = load_and_prepare_data()
    scaled = apply_scalers(tensor, subval, divval)

    original_path = os.path.join(emb_dir, original_inputs_name)
    np.save(original_path, array_to_saved_inputs(tensor.numpy()))
    print(f"Original (non-scaled) inputs saved: {original_path}")

    print(f"Data tensor shape: {tuple(tensor.shape)}")
    print(f"Network input_length: {network_length}, channels: {num_channels}")
    print(f"Found {len(checkpoints)} checkpoint(s)\n")

    summary_lines = []
    all_embed_paths = []

    for ckpt_path in checkpoints:
        flat_dim = parse_flat_dim(ckpt_path)
        header = f"{'=' * 60}\nflat_dim = {flat_dim}  |  {ckpt_path}\n{'=' * 60}"
        print(header)
        summary_lines.append(header + "\n")

        checkpoint = torch.load(ckpt_path, map_location=torch.device(device))
        print(f"Best epoch in checkpoint: {checkpoint.get('epoch', 'n/a')}")
        print(
            f"Checkpoint valid loss: {checkpoint.get('best_valid_loss', 'n/a')}\n"
        )

        model = ModelClass(
            embed_dim=embed_dim,
            flat_dim=flat_dim,
            input_length=network_length,
            input_channels=num_channels,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        # Capture torchsummary to stdout and file
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            _ = summary(model, (num_channels, network_length))
        summ_text = buf.getvalue()
        print(summ_text)
        summary_lines.append(summ_text + "\n")

        t0 = time.time()
        embed, recon_scaled = forward_model(
            model,
            scaled,
            eval_batch_size,
        )
        embed_path = os.path.join(emb_dir, f"embed_flat{flat_dim}.npy")
        np.save(embed_path, embed)
        all_embed_paths.append(embed_path)
        print(f"Embeddings shape: {embed.shape} -> {embed_path}")

        recon_phys = scale_back(recon_scaled, subval, divval)
        recon_path = os.path.join(emb_dir, f"recon_scaledback_flat{flat_dim}.npy")
        np.save(recon_path, array_to_saved_inputs(recon_phys))
        print(
            f"Reconstructions (scaled back) shape: {recon_phys.shape} -> {recon_path}"
        )
        print(f"  Compare with: {original_path}")

        print(f"Forward time: {time.time() - t0:.2f} s\n")

    with open(summ_path, "w") as f:
        f.writelines(summary_lines)
    print(f"Summaries saved to {summ_path}")
    print(f"Embedding files: {', '.join(all_embed_paths)}")


if __name__ == "__main__":
    main()
