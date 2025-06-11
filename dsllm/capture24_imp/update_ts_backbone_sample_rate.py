import yaml
import os

# Paths
config_path = os.path.join(os.path.dirname(__file__), 'config_stage2.yaml')
ts_backbone_path = os.path.join(os.path.dirname(__file__), '../model/ts_backbone.yaml')

# Load config
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
sample_rate = config["original_sampling_rate"] / config["downsample_factor"]

# Load ts_backbone.yaml
with open(ts_backbone_path, 'r') as f:
    ts_config = yaml.safe_load(f)

# Update sample_rate for capture24
if 'capture24' in ts_config:
    ts_config['capture24']['sample_rate'] = sample_rate
else:
    raise KeyError("'capture24' section not found in ts_backbone.yaml")

# Save back
with open(ts_backbone_path, 'w') as f:
    yaml.dump(ts_config, f, sort_keys=False)

print(f"Updated capture24 sample_rate to {sample_rate} in ts_backbone.yaml") 