import importlib
import json
import logging

def load_module(module_name):
    # Load the module
    return importlib.import_module(module_name)

def load_json(json_path, **kwargs):
    # Load the configuration file.
    with open(json_path, 'r') as f:
        return json.load(f)
    
def merge_configs(user, default):
    # Merge user configuration with defaults.
    merged = default.copy()
    merged.update(user)
    return merged
