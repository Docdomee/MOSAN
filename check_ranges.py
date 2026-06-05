import numpy as np
import os

path = r'd:\SERS_Data\Github\cluster\processed_data\generated_datasets\Nature_30_Class_Pretrain'
datasets = ['1D_CNN_data.npy', '2D_GAF_data.npy', '2D_SPECTROGRAM_data.npy', '3D_DYNAMIC_GAF_data.npy']

for d in datasets:
    file_path = os.path.join(path, d)
    if os.path.exists(file_path):
        arr = np.load(file_path, mmap_mode='r')
        # We'll take a sample if it's too large, but mmap + np.min/max is usually fine for range check
        print(f"File: {d}")
        print(f"  dtype: {arr.dtype}")
        print(f"  shape: {arr.shape}")
        # Computing min/max/mean on large memmapped files can be slow, let's sample first 1000 if it's huge
        if arr.size > 10**7:
             sample = arr[:100].copy()
             print(f"  Sample Min: {np.min(sample)}, Sample Max: {np.max(sample)}, Sample Mean: {np.mean(sample)}")
        else:
             print(f"  Min: {np.min(arr)}, Max: {np.max(arr)}, Mean: {np.mean(arr)}")
    else:
        print(f"File: {d} NOT FOUND")
