import argparse
from src.env.reward import compute_reward
from main import load_cfg
    
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    default="config/continuous.yaml",
    help="Path to config YAML (discrete or continuous)",
)
args = parser.parse_args()
cfg = load_cfg(args.config)

# Params for compute_reward
'''
conf_before,
conf_after,
duty,
duration,
cfg,
*,
species=None,
success=False
'''

print(f"Case 1: Bird stays. Reward = {compute_reward(0.9, 0.92, 0.6, 1.2, cfg, species_before='Magpie', success=False)}")

print(f"Case 2: Bird goes. Reward = {compute_reward(0.9, 0.1, 0.6, 1.2, cfg, species_before='Magpie', success=True)}")