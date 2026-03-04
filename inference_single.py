import numpy as np
import argparse
import torch
import torch.nn as nn
import h5py
from tqdm import tqdm
import gc
import os
import json

# Import your sandboxed components for Agent Mode
from models_sandbox import PositionalUNet, AE
from models_format_sandbox import PUNetConfig, AEConfig, LossConfig
from array2h5 import create_abra_file

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_parser():
    """Defines the argument parser for both Fix and Agent modes."""
    parser = argparse.ArgumentParser(description="Inference with Fixed (Baseline) or Agent mode.")
    parser.add_argument('--mode', type=str, choices=['fix', 'agent'], default='fix')
    parser.add_argument('--data_dir', '-d', type=str, default="/home/klz/Data/TIDMAD/")
    parser.add_argument('--denoising_model', '-m', type=str, default='punet')
    parser.add_argument('--file_index', '-i', type=int, default=0)
    
    # Agent Mode Specific Args
    parser.add_argument('--model_cfg', type=str, help="Path to model config JSON")
    parser.add_argument('--loss_cfg', type=str, help="Path to loss config JSON")
    parser.add_argument('--exp_id', type=str, default="default_run")
    parser.add_argument('--model_path', type=str, help="Path to the .pth state_dict")
    return parser

def process_batch(index, inputarr, targetarr, model, args, current_loss_type):
    """
    Handles dimension alignment and output decoding based on LOSS type.
    Aligned with train_engine_sandbox logic.
    """
    # 1. Prepare base tensors
    inputarr = inputarr.astype(np.int16) + 128
    targetarr = targetarr.astype(np.int16) + 128
    input_seq = torch.from_numpy(inputarr)

    # 2. Dynamic Squeezing
    is_linear_based = isinstance(model, AE) or args.denoising_model == "fcnet"
    if is_linear_based:
        if input_seq.dim() == 3:
            input_seq = input_seq.squeeze(1)
    else:
        if input_seq.dim() == 2:
            input_seq = input_seq.unsqueeze(1)

    # 3. Execution & Cast based on Architecture (Input)
    if args.denoising_model == "punet":
        input_seq = input_seq.int().to(DEVICE)
    else:
        input_seq = input_seq.float().to(DEVICE)

    with torch.no_grad():
        output = model(input_seq)
        
        # 4. Decoding based on Loss Type
        if current_loss_type == "smooth_l1":
            # Regression: Direct continuous output
            output_seq = output.detach().cpu().numpy()
        else:
            # Classification: Argmax of logits [Batch, 256, Time]
            output_seq = output.argmax(dim=1).detach().cpu().numpy()
            
    return index, (output_seq - 128).flatten(), (targetarr - 128).flatten()

def main():
    # 1. Parse arguments locally to avoid NameError scope issues
    parser = get_parser()
    args = parser.parse_args()

    # 2. Model Loading Logic
    if args.mode == 'fix':
        model_map = {"punet": "PUNet_0_20.pth", "fcnet": "FCNet_0_20.pth"}
        model_file = model_map.get(args.denoising_model)
        model = torch.load(model_file, map_location=DEVICE, weights_only=False)
        input_size = 40000 
        current_loss_type = "ce"
    
    else:
        # AGENT MODE
        if not args.model_cfg or not args.model_path:
            raise ValueError("Agent mode requires --model_cfg and --model_path")
        
        # Load Loss Type from Config
        loss_path = args.loss_cfg if args.loss_cfg else args.model_cfg.replace("model", "loss")
        with open(loss_path, 'r') as f:
            l_data = json.load(f)
        current_loss_type = l_data.get("loss_type", "ce")

        # Initialize Architecture
        with open(args.model_cfg, 'r') as f:
            m_data = json.load(f)

        if args.denoising_model == "punet":
            m_cfg = PUNetConfig(**m_data)
            model = PositionalUNet(m_cfg, loss_type=current_loss_type).to(DEVICE) 
        else:
            m_cfg = AEConfig(**m_data)
            model = AE(m_cfg, loss_type=current_loss_type).to(DEVICE)
            
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        input_size = m_cfg.segmentation_size

    model.eval()

    # 3. Execution Loop
    fname = f"abra_validation_{str(args.file_index).zfill(4)}.h5"
    fpath = os.path.join(args.data_dir, fname)
    
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Validation data missing at {fpath}")

    with h5py.File(fpath, 'r') as ABRAfile:
        alltrain = np.array(ABRAfile['timeseries']['channel0001']['timeseries'])
        alltarget = np.array(ABRAfile['timeseries']['channel0002']['timeseries'])
        
        train_loader = alltrain.reshape(-1, 1, input_size)
        target_loader = alltarget.reshape(-1, 1, input_size)
        
        dim1 = train_loader.shape[0]
        denoised = np.zeros((dim1, input_size), dtype=np.int8)
        injected = np.zeros((dim1, input_size), dtype=np.int8)

        for i in tqdm(range(dim1), desc=f"Inference ({args.mode})"):
            idx, dn, ij = process_batch(i, train_loader[i:i+1], target_loader[i:i+1], model, args, current_loss_type)
            denoised[idx] = dn
            injected[idx] = ij

    # 4. Save Output
    suffix = f"denoised_{args.denoising_model}" if args.mode == 'fix' else f"denoised_{args.denoising_model}_{args.exp_id}"
    out_name = fpath.replace("abra_validation", f"abra_validation_{suffix}")
    
    create_abra_file(out_name, denoised.flatten().astype(np.int8), injected.flatten().astype(np.int8), indexed=False)
    print(f"Inference complete. Saved to: {out_name}")

if __name__ == "__main__":
    main()