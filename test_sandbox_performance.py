import torch
import torch.nn as nn
from models_sandbox import PositionalUNet, AE
from models_format_sandbox import PUNetConfig, AEConfig

def profile_model(model_cfg):
    """
    Simulates an Agent's proposal to check for OOM and Parameter Count.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Profiling Model: {model_cfg.model_type} ---")
    
    try:
        # 1. Instantiate Model
        if model_cfg.model_type == "punet":
            model = PositionalUNet(model_cfg).to(device)
        else:
            model = AE(model_cfg).to(device)
            
        # 2. Calculate Parameters (Important for the Agent's 'Complexity' score)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params:,}")

        # 3. Dummy Forward Pass (Testing Memory Peak)
        # Shape: [Batch, Length] for ADC integer values
        dummy_input = torch.randint(0, 256, (model_cfg.batch_size, model_cfg.segmentation_size)).to(device)
        
        # Reset peak memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        output = model(dummy_input)
        
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2 # MB
            print(f"Peak GPU Memory: {peak_mem:.2f} MB")
            
        print(f"Output Shape: {output.shape} -> SUCCESS")
        return True

    except torch.cuda.OutOfMemoryError:
        print("CRITICAL: Out of Memory (OOM) detected!")
        return False
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

# Example: Testing a 'Heavy' configuration that might explode
"""
if __name__ == "__main__":
    # Test 1: Standard PUNet
    punet_cfg = PUNetConfig(depth=4, multi=64, segmentation_size=40000)
    profile_model(punet_cfg)
    
    # Test 2: Aggressive AE (High risk of OOM due to Linear layers)
    ae_cfg = AEConfig(latent_dims=[8000, 1000, 100], segmentation_size=40000)
    profile_model(ae_cfg)
"""