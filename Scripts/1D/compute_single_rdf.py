import torch
import numpy as np
import time
from PyISV.features_calc_utils import single_kde_calc
from ase.io import read
from tqdm import tqdm

#### Example: RDF (distance histogram via KDE) using PyISV ###
#
# single_kde_calc expects:
#   - ASE Atoms (positions handled internally)
#   - bins: torch 1D tensor of distance values
#   - bw: kernel bandwidth as torch.Tensor
#   - mic: minimum-image convention for periodic cells

trajectory_path = 'file.xyz'
configurations = read(trajectory_path)

min_dist = 0
max_dist = 4.5
num_bins = 200
bins = torch.linspace(min_dist, max_dist, num_bins)
bw = torch.Tensor([0.05])

periodic_calculation = False
rdfs = []
t0 = time.time()
for conf in tqdm(configurations):
    mic = False
    if periodic_calculation:
        conf.set_pbc(True)
        conf = conf.repeat(2)
        mic = True
    conf_rdf = single_kde_calc(conf, bins, bw, mic=mic)
    rdfs.append(conf_rdf)
elapsed_time = time.time() - t0
print(f'Done, elapsed time: {elapsed_time:.3f} s')

rdfs = np.vstack(rdfs)
np.save('rdfs.npy', rdfs)
