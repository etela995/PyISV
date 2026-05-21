import torch
import numpy as np
import time
from PyISV.features_calc_utils import triple_kde_calc
from ase.io import read
from tqdm import tqdm

#### Example: three partial RDFs for a binary system ###

trajectory_path = 'file.xyz'
configurations = read(trajectory_path)

min_dist = 0
max_dist = 4.5
num_bins = 200
bins = torch.linspace(min_dist, max_dist, num_bins)
bw = torch.Tensor([0.05])

species1 = 'Ti'
species2 = 'O'
periodic_calculation = False

rdfs_s1_s1 = []
rdfs_s2_s2 = []
rdfs_s1_s2 = []
t0 = time.time()
for conf in tqdm(configurations):
    mic = False
    if periodic_calculation:
        conf.set_pbc(True)
        conf = conf.repeat(2)
        mic = True
    s1_s1_rdf, s2_s2_rdf, s1_s2_rdf = triple_kde_calc(
        species1, species2, conf, bins, bw, mic=mic
    )
    rdfs_s1_s1.append(s1_s1_rdf)
    rdfs_s2_s2.append(s2_s2_rdf)
    rdfs_s1_s2.append(s1_s2_rdf)
elapsed_time = time.time() - t0
print(f'Done, elapsed time: {elapsed_time:.3f} s')

rdfs_s1_s1 = np.vstack(rdfs_s1_s1)
rdfs_s2_s2 = np.vstack(rdfs_s2_s2)
rdfs_s1_s2 = np.vstack(rdfs_s1_s2)

collector = np.zeros((len(configurations), 3, num_bins))
collector[:, 0, :] = rdfs_s1_s1
collector[:, 1, :] = rdfs_s2_s2
collector[:, 2, :] = rdfs_s1_s2

np.save(f'{species1}{species2}_triple_rdfs.npy', collector)
