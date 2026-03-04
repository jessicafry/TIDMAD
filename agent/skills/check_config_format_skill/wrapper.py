# skills/check_config_format_skill/wrapper.py

import json
from typing import Dict, Any
import models_format_sandbox as fmt #

def run_skill(sandbox, **kwargs) -> Dict[str, Any]:
    """
    Dynamically extracts the JSON Schema from the Pydantic models in models_format_sandbox.py.
    This ensures the Agent 'reads the manual' before proposing experiments.
    """
    try:
        # Extract schemas for the main configuration categories
        schemas = {
            "PUNetConfig": fmt.PUNetConfig.model_json_schema(),
            "AEConfig": fmt.AEConfig.model_json_schema(),
            "LossConfig": fmt.LossConfig.model_json_schema(),
            "TrainConfig": fmt.TrainConfig.model_json_schema()
        }

        #
        # Summarize constraints for the LLM in a more readable way
        constraints_summary = (
            "1. PUNet: kernel_size must be ODD. Depth vs segmentation_size check active.\n"
            "2. AE: latent_dims must be a list of positive integers.\n"
            "3. Loss: 'focal' requires alpha/gamma. 'smooth_l1' ignores alpha/gamma but uses beta.\n"
            "4. Train: batch_size limit is 128."
        )

        return {
            "status": "success",
            "message": "Configuration schemas retrieved directly from source code.",
            "data": {
                "schemas": schemas,
                "quick_notes": constraints_summary
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to parse Pydantic models: {str(e)}"
        }