# src/utils.py
import json
import os
import numpy as np

#UTILITY FUNCTIONS
# This file contains helper functions for data loading,
# cleaning, and preprocessing used across different scripts.
# Keep this file minimal and generic (no model-specific logic).
#Global seed for reproducibility

SEED = 42
np.random.seed(SEED)

# Data paths
COMPETITION_NAME = 'fds-pokemon-battles-prediction-2025'
DATA_PATH = os.path.join('../input', COMPETITION_NAME)
TRAIN_PATH = os.path.join(DATA_PATH, 'train.jsonl')
TEST_PATH  = os.path.join(DATA_PATH, 'test.jsonl')   

#Load data

def load_jsonl(path):
    """
    Load a .jsonl file (one JSON object per line)
    and return a list of dictionaries.
    """
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def safe_types(tt):
    """
    Ensure that all Pokémon types are valid lowercase strings.
    This prevents errors when a Pokémon has no registered type.
    """
    return [t.lower() if isinstance(t,str) else 'notype' for t in (tt or [])]