#!/usr/bin/env python3
import numpy as np
import argparse
import torch
import torch.nn as nn
import h5py
from tqdm import tqdm
import gc
import os
from network import PositionalUNet, TransformerModel, AE, SimpleWaveNet, RNNSeq2Seq
from array2h5 import create_abra_file
import concurrent.futures

# Use the same DEVICE logic as your training script
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. Parser (Consistent with your Train script)
# ==========================================
parser = argparse.ArgumentParser(description="Run inference over a single validation file.")

parser.add_argument('--data_dir', '-d', type=str, default="/home/klz/Data/TIDMAD/", help='Directory where the data is stored.')
parser.add_argument('--denoising_model', '-m', type=str, default='punet', help='Model to use [fcnet/punet/transformer/wavenet/rnn].')
parser.add_argument('--file_index', '-i', type=int, default=0, help="Index for the validation file (abra_validation_XXXX.h5)")
parser.add_argument('--parallel', '-p', action='store_true', help='Use multiprocessing for pre/post processing.')

args = parser.parse_args()

# Match the input_size used in your training script for consistency
input_size = 40000 
if args.denoising_model == "transformer":
    input_size = 20000
batchsize = 1

# ==========================================
# 2. Core Processing Logic
# ==========================================

def process_batch(index, inputarr, targetarr, model, args):
    """
    Handles the +128 offset, model forward pass, and -128 offset.
    """
    # Offset logic from baseline
    inputarr = inputarr.astype(np.int16) + 128
    targetarr = targetarr.astype(np.int16) + 128
    
    input_seq = torch.from_numpy(inputarr)

    if args.denoising_model == "fcnet":
        input_seq = input_seq.float().to(DEVICE)
        output_seq = model(input_seq).detach().cpu().numpy()
    else:
        # Classification models use long() and argmax()
        input_seq = input_seq.long().to(DEVICE)
        output_seq = model(input_seq).argmax(dim=1).detach().cpu().numpy()
        
    # Return flattened data with offset removed
    return index, (output_seq - 128).flatten(), (targetarr - 128).flatten()

def main():
    # Construct filename based on index
    fname = f"abra_validation_{str(args.file_index).zfill(4)}.h5"
    fpath = os.path.join(args.data_dir, fname)
    
    if not os.path.exists(fpath):
        print(f"Error: File {fpath} not found.")
        return

    # Load Model (Fixed for PyTorch 2.6 weights_only issue)
    model_map = {
        "punet": "PUNet_0_20.pth",
        "fcnet": "FCNet_0_20.pth",
        "transformer": "Transformer_0_20.pth",
        "wavenet": "WaveNet_0_20.pth",
        "rnn": "RNN_0_20.pth"
    }
    
    model_file = model_map.get(args.denoising_model)
    print(f"Loading {args.denoising_model} from {model_file}...")
    
    # Use weights_only=False to support your AE class from network.py
    model = torch.load(model_file, map_location=DEVICE, weights_only=False)
    model.eval()

    # Read and reshape validation data
    with h5py.File(fpath, 'r') as ABRAfile:
        alltrain = np.array(ABRAfile['timeseries']['channel0001']['timeseries'])
        alltarget = np.array(ABRAfile['timeseries']['channel0002']['timeseries'])
        
        max_index = 2000000000
        # Reshape to match the training segmentation: [-1, 1, 40000]
        train_loader = alltrain[:max_index].reshape(-1, batchsize, input_size)
        target_loader = alltarget[:max_index].reshape(-1, batchsize, input_size)
        
        dim1 = train_loader.shape[0]
        denoised = np.zeros((dim1, input_size), dtype=np.int8)
        injected = np.zeros((dim1, input_size), dtype=np.int8)

        if args.parallel:
            workers = 8
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                tasks = []
                for i in range(dim1):
                    in_batch = train_loader[i:i+1]
                    tg_batch = target_loader[i:i+1]
                    tasks.append(executor.submit(process_batch, i, in_batch, tg_batch, model, args))
                
                for future in tqdm(concurrent.futures.as_completed(tasks), total=len(tasks), desc="Parallel Inference"):
                    idx, dn, ij = future.result()
                    denoised[idx] = dn
                    injected[idx] = ij
        else:
            for i in tqdm(range(dim1), desc="Sequential Inference"):
                in_batch = train_loader[i:i+1]
                tg_batch = target_loader[i:i+1]
                idx, dn, ij = process_batch(i, in_batch, tg_batch, model, args)
                denoised[idx] = dn
                injected[idx] = ij

    # Flatten for creation of H5 file
    denoised_final = denoised.flatten().astype(np.int8)
    injected_final = injected.flatten().astype(np.int8)

    # Save output
    out_name = fpath.replace("abra_validation", f"abra_validation_denoised_{args.denoising_model}")
    create_abra_file(out_name, denoised_final, injected_final, indexed=False)
    print(f"Inference complete. Result saved to: {out_name}")

if __name__ == "__main__":
    main()