from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Literal, Optional, Union

# ==========================================
# 1. Base Configuration
# ==========================================

class BaseConfig(BaseModel):
    """General config"""
    segmentation_size: int = Field(default=40000, ge=1000, description="Input time series length")
    batch_size: int = Field(default=1, ge=1)
    
# ==========================================
# 2. PUNet Configuration (Convolutional)
# ==========================================
class PUNetConfig(BaseConfig):
    model_type: Literal["punet"] = "punet"
    multi: int = Field(default=40, ge=16, le=128, description="Base channel multiplier")
    depth: int = Field(default=4, ge=1, le=5, description="Number of downsampling steps")
    bilinear: bool = Field(default=True, description="Whether to use bilinear upsampling or conv transpose")
    pe_factor: float = Field(default=1.0, ge=0.0, le=10.0, description="Positional encoding weight")
    kernel_size: int = Field(default=9, ge=3, le=15, description="Convolutional kernel size")
    embedding_dim: int = Field(default=32, ge=8, le=256, description="Latent space dimension for ADC values")

    @field_validator('kernel_size')
    @classmethod
    def kernel_must_be_odd(cls, v: int) -> int:
        """Ensures symmetric padding is possible."""
        if v % 2 == 0:
            raise ValueError('kernel_size must be odd for symmetric padding')
        return v

    @model_validator(mode='after')
    def check_dimension_reduction(self) -> 'PUNetConfig':
        """
        Physical/Architecture Constraint:
        Ensures segmentation_size is large enough to sustain the chosen depth.
        With a stride of 4, the size reduces by 4^depth.
        """
        reduction_factor = 4 ** self.depth
        if self.segmentation_size < reduction_factor:
            raise ValueError(
                f"segmentation_size ({self.segmentation_size}) is too small for "
                f"depth ({self.depth}). Minimum required: {reduction_factor}"
            )
        
        # Optional: Check if size is a multiple of the reduction factor for clean division
        if self.segmentation_size % reduction_factor != 0:
            # We don't necessarily raise an error because F.pad handles it, 
            # but it's good practice to log or warn.
            pass
            
        return self

# ==========================================
# 3. AE Configuration (Fully Connected)
# ==========================================
class AEConfig(BaseConfig):
    """
    Configuration for Fully Connected AutoEncoder.
    Designed for global feature compression and noise filtering.
    """
    model_type: Literal["ae"] = "ae"
    # Agent can modify the depth and width by changing this list
    latent_dims: List[int] = Field(
        default=[4000, 400, 40], 
        min_length=1, 
        max_length=6,
        description="List of hidden layer dimensions for the encoder"
    )
    dropout: float = Field(default=0.0, ge=0.0, le=0.5)

    @field_validator('latent_dims')
    @classmethod
    def check_dimensions(cls, v: List[int]) -> List[int]:
        if any(d <= 0 for d in v):
            raise ValueError("All hidden dimensions must be positive integers.")
        return v
    
# ==========================================
# 4. Global Model Registry
# ==========================================

# Union type for the Agent to choose from
ModelConfigUnion = Union[PUNetConfig, AEConfig]

def get_config_class(model_type: str):
    """Helper for the Orchestrator to map strings to Pydantic classes."""
    mapping = {
        "punet": PUNetConfig,
        "fcnet": AEConfig
    }
    return mapping.get(model_type)

# ==========================================
# 5. Loss Configs
# ==========================================

class LossConfig(BaseModel):
    """
    Configuration for the Denoising Scoring Functions (Loss).
    Strictly validates parameters based on the chosen loss_type.
    """
    loss_type: Literal["focal", "focal_cw", "ce", "smooth_l1"] = "focal"
    
    # Parameters for Focal / Focal_CW
    alpha: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)
    gamma: Optional[float] = Field(default=2.0, ge=0.0, le=5.0)
    
    # Parameter for SmoothL1
    beta: Optional[float] = Field(default=1.0, ge=0.1, le=10.0)
    
    reduction: Literal["mean", "sum"] = "mean"
    use_class_weights: bool = Field(default=False)

    @model_validator(mode='after')
    def enforce_parameter_consistency(self) -> 'LossConfig':
        """
        Ensures that only relevant parameters are active for the selected loss_type.
        This prevents the Agent from 'hallucinating' cross-parameter optimizations.
        """
        if self.loss_type in ["focal", "focal_cw"]:
            # SmoothL1 parameter is irrelevant here
            self.beta = None 
        
        elif self.loss_type == "smooth_l1":
            # Focal parameters are irrelevant here
            self.alpha = None
            self.gamma = None
            # Regression usually doesn't use class weights in the same way
            self.use_class_weights = False
            
        elif self.loss_type == "ce":
            # CrossEntropy is the baseline, no special hyperparams needed
            self.alpha = None
            self.gamma = None
            self.beta = None
            
        return self
    
# ==========================================
# 6. Training Config
# ==========================================

class TrainConfig(BaseModel):
    """
    Configuration for the training execution.
    Agent can optimize learning rate, optimizer type, and epochs.
    """
    lr: float = Field(default=1e-4, ge=1e-6, le=1e-1)
    epochs: int = Field(default=10, ge=1, le=100)
    optimizer_type: Literal["adam", "adamw", "sgd"] = "adamw"
    weight_decay: float = Field(default=1e-5, ge=0, le=1e-1)
    device: str = "cuda" # or "cpu"