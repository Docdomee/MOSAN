
import os
import random
import numpy as np
import tensorflow as tf
from typing import Optional
from ui_logger import log

def set_global_seed(seed: int = 42) -> int:
    """
    Sets the random seed for all major libraries to ensure reproducibility.
    
    Args:
        seed: The integer seed to use. Defaults to 42.
        
    Returns:
        int: The seed that was set.
    """
    log(f"[Reproducibility] locking experimentation seed to: {seed}")
    
    # 1. Python core
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    
    # 2. NumPy
    np.random.seed(seed)
    
    # 3. TensorFlow
    tf.random.set_seed(seed)
    
    # 4. Optional: PyTorch (if installed in future)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
        
    return seed

def get_deterministic_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Returns a dedicated numpy random generator properly seeded.
    Use this instead of np.random for localized randomness.
    """
    if seed is None:
        seed = 42
    return np.random.default_rng(seed)
