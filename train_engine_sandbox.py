import os
import gc
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import h5py
import itertools
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

# Import your sandboxed components
from models_sandbox import PositionalUNet, AE
from models_format_sandbox import PUNetConfig, AEConfig, TrainConfig, LossConfig
from loss_models_sandbox import get_criterion

# ==========================================
# 1. Reused Dataset Logic from Original Script
# ==========================================

class TIDMADDataset(Dataset):
    """
    Revised TIDMAD dataset for Agent experiments.
    Ensures filelist is handled as a list and batching is offloaded to DataLoader.
    """
    def __init__(self, fpath: str, fname_list: list, segmentation_size: int, sample_size: int = 20):
        self.filepath = fpath
        # Ensure we are dealing with a list
        self.filelist = fname_list if isinstance(fname_list, list) else [fname_list]
        self.seg_size = segmentation_size
        self.sample_size = sample_size
        
        self.idict = {}
        self.tdict = {}
        self.class_count = torch.ones(256)

        # Store indexed segments as (filename, row_index)
        self.train_events = self.pull_event_from_dir(self.filelist)
        self.size = len(self.train_events)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        filename, row_idx = self.train_events[idx]
        
        # Access pre-processed segments in memory
        input_data = self.idict[filename][row_idx]
        target_data = self.tdict[filename][row_idx]

        # Shift ADC range [ -128, 127 ] -> [ 0, 255 ] and convert to int16
        # Returning shape: [segmentation_size]
        return (input_data.astype(np.int16) + 128), (target_data.astype(np.int16) + 128)

    def get_class_weight(self):
        """Calculates frequency-based weights for the 256 ADC classes."""
        weights = self.class_count.sum() / (len(self.class_count) * self.class_count)
        weights = weights / weights.sum() * len(self.class_count)
        return weights

    def pull_event_from_dir(self, filelist):
        evlist = []
        max_points = 2000000000 
        
        for filename in tqdm(filelist, desc="Indexing H5 Data"):
            file_path = os.path.join(self.filepath, filename)
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} not found. Skipping.")
                continue

            with h5py.File(file_path, 'r') as f:
                # Load into memory as int8 to save space
                alltrain = np.array(f['timeseries']['channel0001']['timeseries']).astype(np.int8)
                alltarget = np.array(f['timeseries']['channel0002']['timeseries']).astype(np.int16)
                
                # Logic alignment with original script: 
                # We only keep one random segment from every block of 'sample_size'
                total_len = min(len(alltrain), max_points)
                num_segments = total_len // (self.sample_size * self.seg_size)
                
                # Pick one random offset per file to keep data diversity
                random_offset = np.random.randint(0, self.sample_size)
                
                # Reshape and Slice: [num_segments, sample_size, seg_size] -> [num_segments, seg_size]
                reshaped_in = alltrain[:num_segments * self.sample_size * self.seg_size].reshape(
                    num_segments, self.sample_size, self.seg_size
                )[:, random_offset, :]
                
                reshaped_target = alltarget[:num_segments * self.sample_size * self.seg_size].reshape(
                    num_segments, self.sample_size, self.seg_size
                )[:, random_offset, :].astype(np.int8)

                self.idict[filename] = reshaped_in
                self.tdict[filename] = reshaped_target
                
                # Update class distribution for LossConfig's use_class_weights
                self.class_count += torch.Tensor(np.bincount(alltarget + 128, minlength=256))
                
                # Record (filename, row_index) for __getitem__
                for i in range(num_segments):
                    evlist.append((filename, i))
                
                del alltrain, alltarget
                gc.collect()
        
        return evlist
# ==========================================
# 2. Training Engine Logic
# ==========================================

def run_experiment(model_cfg, train_cfg: TrainConfig, loss_cfg: LossConfig, data_loader: DataLoader):
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")
    
    # Model Initialization
    if isinstance(model_cfg, PUNetConfig):
        model = PositionalUNet(model_cfg).to(device)
    elif isinstance(model_cfg, AEConfig):
        model = AE(model_cfg).to(device)
    else:
        raise ValueError("Invalid model configuration type provided.")

    # Criterion Setup
    class_weights = None
    if loss_cfg.use_class_weights:
        class_weights = data_loader.dataset.get_class_weight().to(device)
    
    criterion = get_criterion(loss_cfg, class_weights)

    # Optimizer Setup
    if train_cfg.optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    elif train_cfg.optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=train_cfg.lr)

    # Training Loop
    history = []
    for ep in range(train_cfg.epochs):
        model.train()
        batch_losses = []
        for i, (input_batch, target_batch) in enumerate(tqdm(data_loader, desc=f"Epoch {ep}")):
            # Move entire batch to device
            input_seq = input_batch.to(device)   # Shape: [batch_size, seg_size]
            target_seq = target_batch.to(device) # Shape: [batch_size, seg_size]

            # Type conversion and channel unsqueeze if necessary
            if model_cfg.model_type != "ae":
                # PUNet expects [Batch, Time] (integers) which it then embeds to [Batch, Time, Emb]
                # Then transposes to [Batch, Emb, Time] internally.
                input_seq = input_seq.int()
                target_seq = target_seq.long()
            else:
                input_seq = input_seq.float()
                target_seq = target_seq.float()

            optimizer.zero_grad()
            output = model(input_seq)
            
            # Loss calculation
            loss = criterion(output, target_seq)
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.item())
            
        avg_epoch_loss = np.mean(batch_losses)
        history.append(float(avg_epoch_loss))
        print(f"Epoch {ep} | Average Loss: {avg_epoch_loss:.6f}")
        
    # Result Summary
    summary = {
        "final_loss": history[-1],
        "loss_history": history,
        "model_params": sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    exp_id = getattr(train_cfg, 'experiment_id', 'default_run') 
    save_path = f"cached_models/model_{model_cfg.model_type}_{exp_id}_agent.pth"

    # Save and Cleanup
    torch.save(model.state_dict(), save_path)
    del model, optimizer, criterion
    torch.cuda.empty_cache()
    gc.collect()

    return summary

# ==========================================
# 3. Main Entry Point (Parser)
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="TIDMAD Training Engine Sandbox with Pydantic Support")
    parser.add_argument("--model_cfg", type=str, required=True, help="JSON file for model config")
    parser.add_argument("--train_cfg", type=str, required=True, help="JSON file for train config")
    parser.add_argument("--loss_cfg", type=str, required=True, help="JSON file for loss config")
    parser.add_argument("--data_dir", type=str, default="/home/klz/Data/TIDMAD/")
    parser.add_argument("--file_index", type=int, default=0)

    args = parser.parse_args()

    # Load JSONs
    with open(args.model_cfg, 'r') as f:
        m_data = json.load(f)
    with open(args.train_cfg, 'r') as f:
        t_data = json.load(f)
    with open(args.loss_cfg, 'r') as f:
        l_data = json.load(f)

    # Convert to Pydantic Objects (Triggers Validation)
    if m_data.get("model_type") == "punet":
        model_cfg = PUNetConfig(**m_data)
    elif m_data.get("model_type") == "ae":
        model_cfg = AEConfig(**m_data)
    else:
        raise ValueError("Unknown model_type in config.")

    train_cfg = TrainConfig(**t_data)
    loss_cfg = LossConfig(**l_data)

    # Inside train_engine_sandbox.py -> main()

    # 1. Prepare the file list correctly
    fname = f"abra_training_{str(args.file_index).zfill(4)}.h5"
    file_list = [fname] # Wrap in a list to satisfy the Dataset requirement

    # 2. Initialize Dataset
    dataset = TIDMADDataset(
        fpath=args.data_dir, 
        fname_list=file_list, 
        segmentation_size=model_cfg.segmentation_size
    )

    # 3. Initialize DataLoader with Agent's preferred batch_size
    train_loader = DataLoader(
        dataset, 
        batch_size=train_cfg.batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=4, # Multiprocessing for speed
        pin_memory=True # Faster transfer to GPU
    )

    # Run Execution
    print(f"--- Experiment Starting: {model_cfg.model_type} ---")
    results = run_experiment(model_cfg, train_cfg, loss_cfg, train_loader)
    
    
    # Output results for Agent to read from stdout or file
    exp_id = getattr(train_cfg, 'experiment_id', 'default_run')
    with open(f"results/experiment_results_{model_cfg.model_type}_{exp_id}.json", "w") as rf:
        json.dump(results, rf, indent=4)
    print("Experiment Completed successfully.")

if __name__ == "__main__":
    main()