import numpy as np
import argparse
import torch
import torch.nn as nn
import h5py
from tqdm import tqdm
import gc
import os
import json
import concurrent.futures

# Import your sandboxed components for Agent Mode
from models_sandbox import PositionalUNet, AE
from models_format_sandbox import PUNetConfig, AEConfig
from array2h5 import create_abra_file

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. Parser (Updated for Dual Mode)
# ==========================================
parser = argparse.ArgumentParser(description="Inference with Fixed (Baseline) or Agent mode.")

parser.add_argument('--mode', type=str, choices=['fix', 'agent'], default='fix', help='Run baseline (fix) or Agent exploration (agent).')
parser.add_argument('--data_dir', '-d', type=str, default="/home/klz/Data/TIDMAD/")
parser.add_argument('--denoising_model', '-m', type=str, default='punet')
parser.add_argument('--file_index', '-i', type=int, default=0)
parser.add_argument('--parallel', '-p', action='store_true')

# Agent Mode Specific Args
parser.add_argument('--model_cfg', type=str, help="Required for Agent mode: Path to model config JSON")
parser.add_argument('--exp_id', type=str, default="default_run", help="Experiment ID from Train Engine")
parser.add_argument('--model_path', type=str, help="Path to the .pth state_dict from Train Engine")

args = parser.parse_args()

def process_batch(index, inputarr, targetarr, model, args):
    """
    Handles dimension alignment dynamically for different model types.
    """
    # 1. Prepare base tensors
    inputarr = inputarr.astype(np.int16) + 128
    targetarr = targetarr.astype(np.int16) + 128
    
    # input_seq from main loop is likely [1, 1, Time]
    input_seq = torch.from_numpy(inputarr)

    # 2. Dynamic Squeezing/Unsqueezing based on Architecture
    # Check if the model starts with an Embedding layer (like your new PUNet)
    has_embedding = hasattr(model, 'embedding') and isinstance(model.embedding, nn.Embedding)
    # Check if it's an AE or FCNet (Linear layers)
    is_linear_based = isinstance(model, AE) or args.denoising_model == "fcnet"

    if has_embedding or is_linear_based:
        # These models expect [Batch, Time] (2D)
        # Squeeze out the extra 'channel' dimension if present: [1, 1, 40000] -> [1, 40000]
        if input_seq.dim() == 3:
            input_seq = input_seq.squeeze(1)
    else:
        # Baseline models or Conv-first models usually expect [Batch, 1, Time] (3D)
        # Ensure it has 3 dimensions
        if input_seq.dim() == 2:
            input_seq = input_seq.unsqueeze(1)

    # 3. Execution
    if is_linear_based:
        input_seq = input_seq.float().to(DEVICE)
        output_seq = model(input_seq).detach().cpu().numpy()
    else:
        # For Classification (PUNet etc.)
        input_seq = input_seq.long().to(DEVICE)
        output_seq = model(input_seq).argmax(dim=1).detach().cpu().numpy()
    return index, (output_seq - 128).flatten(), (targetarr - 128).flatten()

def main():
    # 1. Model Loading Logic
    if args.mode == 'fix':
        # --- Original Baseline Logic ---
        model_map = {
            "punet": "PUNet_0_20.pth",
            "fcnet": "FCNet_0_20.pth",
            "transformer": "Transformer_0_20.pth"
        }
        model_file = model_map.get(args.denoising_model)
        print(f"FIX MODE: Loading baseline {args.denoising_model} from {model_file}...")
        # Baseline usually saves the whole object
        model = torch.load(model_file, map_location=DEVICE, weights_only=False)
        input_size = 40000 
    
    else:
        # --- Agent Sandbox Logic ---
        print(f"AGENT MODE: ID {args.exp_id} | Model {args.denoising_model}")
        if not args.model_cfg or not args.model_path:
            raise ValueError("Agent mode requires --model_cfg and --model_path")
        
        # Load Config to get dynamic architecture
        with open(args.model_cfg, 'r') as f:
            m_data = json.load(f)
        
        if args.denoising_model == "punet":
            m_cfg = PUNetConfig(**m_data)
            model = PositionalUNet(m_cfg).to(DEVICE)
        else:
            m_cfg = AEConfig(**m_data)
            model = AE(m_cfg).to(device)
            
        # Load state_dict (weight only) from Train Engine
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        input_size = m_cfg.segmentation_size

    model.eval()

    # 2. Filename and Path Setup
    fname = f"abra_validation_{str(args.file_index).zfill(4)}.h5"
    fpath = os.path.join(args.data_dir, fname)
    
    # 3. Execution (Same as before but with dynamic input_size)
    with h5py.File(fpath, 'r') as ABRAfile:
        alltrain = np.array(ABRAfile['timeseries']['channel0001']['timeseries'])
        alltarget = np.array(ABRAfile['timeseries']['channel0002']['timeseries'])
        
        # Batchsize 1 is standard for validation here
        train_loader = alltrain.reshape(-1, 1, input_size)
        target_loader = alltarget.reshape(-1, 1, input_size)
        
        dim1 = train_loader.shape[0]
        denoised = np.zeros((dim1, input_size), dtype=np.int8)
        injected = np.zeros((dim1, input_size), dtype=np.int8)

        # ... (Sequential or Parallel processing loop remains the same) ...
        for i in tqdm(range(dim1), desc=f"Inference ({args.mode})"):
            idx, dn, ij = process_batch(i, train_loader[i:i+1], target_loader[i:i+1], model, args)
            denoised[idx] = dn
            injected[idx] = ij

    # 4. Save Output with Mode-Specific Naming
    if args.mode == 'fix':
        out_name = fpath.replace("abra_validation", f"abra_validation_denoised_{args.denoising_model}")
    else:
        # Agent output includes Exp ID to avoid overwriting baseline or other exps
        out_name = fpath.replace("abra_validation", f"abra_validation_denoised_{args.denoising_model}_{args.exp_id}")
    
    create_abra_file(out_name, denoised.flatten().astype(np.int8), injected.flatten().astype(np.int8), indexed=False)
    print(f"Inference complete. Saved to: {out_name}")

if __name__ == "__main__":
    main()