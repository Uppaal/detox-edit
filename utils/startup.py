import os
import torch
import random
import numpy as np
import configparser
from transformers import set_seed

def main(config_filename):
    config = configparser.ConfigParser()
    config_path = f'configs/{config_filename}'

    assert os.path.exists(config_path), f'Config file not found at {config_path}'
    config.read(config_path)

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        print(f'CUDA VISIBLE DEVICES = {os.environ["CUDA_VISIBLE_DEVICES"]}')
    else:
        gpu_num = config.get('GPUs', 'cuda_visible_devices')
        print(f'No CUDA_VISIBLE_DEVICES found. Setting to {gpu_num}.')
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"

    project_root = config.get('Filepaths', 'project_root')
    dataset_dir = config.get('Filepaths', 'dataset_dir')
    hf_home = config.get('Filepaths', 'hf_home')
    seed = config.getint('GPUs', 'seed')
    hf_token = config.get('Model', 'hf_token')

    # Update or set environment variables
    python_path = os.environ.get("PYTHONPATH", "")
    os.environ["PROJECT_ROOT"] = project_root
    os.environ["PYTHONPATH"] = f"{project_root}:{python_path}"
    os.environ["DATASET_DIR"] = dataset_dir
    os.environ["HF_HOME"] = hf_home
    os.environ["HF_TOKEN"] = hf_token

    print("PROJECT_ROOT:", os.environ.get("PROJECT_ROOT"))
    print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
    print("HF_HOME:", os.environ.get("HF_HOME"))

    set_seed(seed)            # Huggingface
    random.seed(seed)         # Python
    np.random.seed(seed)      # Numpy
    torch.manual_seed(seed)   # PyTorch

    return config
