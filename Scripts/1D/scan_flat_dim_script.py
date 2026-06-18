"""Scan FlexibleAutoencoder over flat_dim values on data loaded from .npy.

Edit the PARAMETERS section, then run from Scripts/1D/:

    python scan_flat_dim_script.py

Data may have a different number of bins than the network expects. Use
length_mode='resample' to linearly remap every curve to input_length (default
200), or length_mode='native' to set the network input_length from your data.
"""
import csv
import os
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from PyISV.network_flex import FlexibleAutoencoder
from PyISV.network_transformer import TransformerAutoencoder
# from PyISV.network_transformer import TransformerAutoencoder  # swap for transformer comparison
from PyISV.train_utils import (
    Dataset,
    EarlyStopping,
    MSELoss,
    SaveBestModel,
    resample_descriptors,
)

##########################
####### PARAMETERS #######
##########################

input_path = "path_to_inputs.npy"
input_dimensionality = 1  # 1 -> (N, L) or (N, C, L)

# How to handle bin count different from the network:
#   "resample" -> map last axis to input_length via linear interpolation
#   "native"   -> use data.shape[-1] as network input_length (no resampling)
length_mode = "resample"
input_length = 200  # used when length_mode == "resample"

flat_dims_to_scan = [1, 2, 4, 8, 16, 21]
embed_dim = 2
input_channels = None  # None: infer from data; or set explicitly (e.g. 1, 3)

train_fraction = 0.8
batch_size = 64
seed = 7352143264209594346
device = "cpu"
norm_mode = "minmax"

min_num_epochs = 30
max_num_epochs = 150
stopper_patience = 8
stopper_delta = 0.00005
learning_rate = 5e-3

output_dir = "flat_dim_scan"
results_csv = "flat_dim_scan_results.csv"

# FlexibleAutoencoder (conv) or TransformerAutoencoder for architecture comparison
ModelClass = FlexibleAutoencoder


def prepare_tensor_data(raw, num_channels):
    """Return float tensor shaped (N, C, L) for the flexible autoencoder."""
    if raw.ndim == 2:
        if num_channels != 1:
            raise ValueError(
                f"Data is (N, L) but input_channels={num_channels}; "
                "use (N, C, L) or set input_channels=1"
            )
        return torch.tensor(raw, dtype=torch.double).unsqueeze(1)
    if raw.ndim == 3:
        if num_channels is not None and raw.shape[1] != num_channels:
            raise ValueError(
                f"Data has {raw.shape[1]} channels but input_channels={num_channels}"
            )
        return torch.tensor(raw, dtype=torch.double)
    raise ValueError(f"Unsupported data shape {raw.shape}")


def make_loaders(raw, train_fraction, batch_size, norm_mode):
    num_channels = 1 if raw.ndim == 2 else raw.shape[1]
    full = Dataset(
        prepare_tensor_data(raw, num_channels),
        prepare_tensor_data(raw, num_channels),
        norm_inputs=True,
        norm_targets=True,
        norm_mode=norm_mode,
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        full.inputs,
        full.targets,
        train_size=train_fraction,
        shuffle=True,
        random_state=seed,
    )
    train_loader = DataLoader(
        Dataset(x_train, y_train, norm_inputs=False, norm_targets=False),
        shuffle=True,
        batch_size=batch_size,
        drop_last=True,
    )
    valid_loader = DataLoader(
        Dataset(x_valid, y_valid, norm_inputs=False, norm_targets=False),
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
    )
    return train_loader, valid_loader, num_channels


def train_flat_dim(
    flat_dim,
    input_length,
    num_channels,
    train_loader,
    valid_loader,
):
    torch.manual_seed(seed)
    model = ModelClass(
        embed_dim=embed_dim,
        flat_dim=flat_dim,
        input_length=input_length,
        input_channels=num_channels,
    ).to(device)

    loss_fn = MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    saver = SaveBestModel(best_model_name=f"best_model_flat{flat_dim}")
    stopper = EarlyStopping(patience=stopper_patience, min_delta=stopper_delta)

    epochs_run = 0
    for epoch in range(max_num_epochs):
        epochs_run = epoch + 1
        model.train()
        train_loss = 0.0
        n_batches = 0
        for x, _ in train_loader:
            n_batches += 1
            x = x.float().to(device)
            output, _ = model(x)
            loss = loss_fn(output, x)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
        train_loss /= n_batches

        model.eval()
        valid_loss = 0.0
        n_valid = 0
        with torch.no_grad():
            for x_valid, _ in valid_loader:
                n_valid += 1
                x_valid = x_valid.float().to(device)
                output_valid, _ = model(x_valid)
                valid_loss += loss_fn(output_valid, x_valid).item()
        valid_loss /= n_valid

        saver(valid_loss, train_loss, epoch, model, optimizer)
        if epoch >= min_num_epochs and stopper(valid_loss):
            break

    os.makedirs(output_dir, exist_ok=True)
    dest = os.path.join(output_dir, f"best_model_flat{flat_dim}.pth")
    local_ckpt = f"best_model_flat{flat_dim}.pth"
    if os.path.exists(local_ckpt):
        os.replace(local_ckpt, dest)

    return {
        "flat_dim": flat_dim,
        "best_valid_loss": saver.best_valid_loss,
        "best_train_loss": saver.best_train_loss,
        "best_epoch": saver.best_epoch,
        "epochs_run": epochs_run,
        "linear_params": 128 * flat_dim * embed_dim * 2,
    }


def main():
    os.makedirs(output_dir, exist_ok=True)
    raw = np.load(input_path)
    native_length = raw.shape[-1]

    if length_mode == "resample":
        network_length = input_length
        if native_length != network_length:
            print(
                f"Resampling descriptors: {native_length} bins -> {network_length} bins"
            )
            raw = resample_descriptors(raw, network_length)
        else:
            print(f"Data already has {network_length} bins; no resampling.")
    elif length_mode == "native":
        network_length = native_length
        print(f"Using native length: {network_length} bins (no resampling).")
    else:
        raise ValueError('length_mode must be "resample" or "native"')

    if len(raw.shape) > input_dimensionality + 1:
        num_channels = raw.shape[1] if input_channels is None else input_channels
    else:
        num_channels = 1 if input_channels is None else input_channels

    print(f"Data shape after prep: {raw.shape}")
    print(f"Network input_length: {network_length}")
    print(f"Channels: {num_channels}")
    print(f"Scanning flat_dim in {flat_dims_to_scan}\n")

    train_loader, valid_loader, num_channels = make_loaders(
        raw, train_fraction, batch_size, norm_mode
    )

    # Scalers for later evaluation (same normalization as training)
    full_ds = Dataset(
        prepare_tensor_data(raw, num_channels),
        prepare_tensor_data(raw, num_channels),
        norm_inputs=True,
        norm_targets=True,
        norm_mode=norm_mode,
    )
    np.save(os.path.join(output_dir, "input_scaler_subval.npy"), full_ds.subval_inputs.numpy())
    np.save(os.path.join(output_dir, "input_scaler_divval.npy"), full_ds.divval_inputs.numpy())
    np.save(
        os.path.join(output_dir, "scan_config.npy"),
        {
            "input_length": network_length,
            "native_length": native_length,
            "length_mode": length_mode,
            "embed_dim": embed_dim,
            "num_channels": num_channels,
            "norm_mode": norm_mode,
        },
    )

    results = []
    t_scan = time.time()
    for flat_dim in flat_dims_to_scan:
        print(f"--- flat_dim = {flat_dim} ---")
        t0 = time.time()
        row = train_flat_dim(
            flat_dim,
            network_length,
            num_channels,
            train_loader,
            valid_loader,
        )
        row["elapsed_s"] = time.time() - t0
        row["input_length"] = network_length
        row["native_length"] = native_length
        row["length_mode"] = length_mode
        results.append(row)
        print(
            f"  best valid loss = {row['best_valid_loss']:.6f} "
            f"(epoch {row['best_epoch']}, {row['elapsed_s']:.1f} s)\n"
        )

    csv_path = os.path.join(output_dir, results_csv)
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    best = min(results, key=lambda r: r["best_valid_loss"])
    print(f"Scan finished in {time.time() - t_scan:.1f} s")
    print(f"Results written to {csv_path}")
    print(
        f"Best flat_dim = {best['flat_dim']} "
        f"(valid loss {best['best_valid_loss']:.6f})"
    )


if __name__ == "__main__":
    main()
