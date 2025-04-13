import sys
import os
# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from ablation_study import DiffSearchAblation
from models.diffusion import Model as DiffusionModel
from datasets import get_dataloader
from config import get_config

# Initialize model and dataloader
config = get_config()
model = DiffusionModel(config)
dataloader = get_dataloader(batch_size=16)

# Run ablation study
ablation = DiffSearchAblation(model, dataloader)
ablation.run_ablation()
ablation.visualize_weights_evolution()
ablation.analyze_components_importance() 