
import threading
from typing import Any, Dict, Optional, Union

class SharedStateManager:
    """
    A thread-safe proxy that ensures all agents read/write directly 
    to the authoritative session state.
    Mimics a dictionary interface to be compatible with existing code.
    """
    def __init__(self, session_state_ref: Any):
        self._state = session_state_ref
        self._lock = threading.Lock()

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            if isinstance(self._state, dict):
                return self._state[key]
            return getattr(self._state, key)

    def __setitem__(self, key: str, value: Any):
        with self._lock:
            if isinstance(self._state, dict):
                self._state[key] = value
            else:
                setattr(self._state, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if isinstance(self._state, dict):
                return self._state.get(key, default)
            return getattr(self._state, key, default)

    def pop(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if isinstance(self._state, dict):
                return self._state.pop(key, default)
            elif hasattr(self._state, key):
                val = getattr(self._state, key)
                delattr(self._state, key)
                return val
            return default

    def update(self, other: Union[Dict, 'SharedStateManager']):
        with self._lock:
            if isinstance(other, SharedStateManager):
                # accessing private member of other is safe if same class
                other_dict = other._state if isinstance(other._state, dict) else other._state.__dict__
            else:
                other_dict = other

            if isinstance(self._state, dict):
                self._state.update(other_dict)
            else:
                for k, v in other_dict.items():
                    setattr(self._state, k, v)
    
    def items(self):
        with self._lock:
             if isinstance(self._state, dict):
                 return list(self._state.items())
             return list(self._state.__dict__.items())

    def keys(self):
        with self._lock:
             if isinstance(self._state, dict):
                 return list(self._state.keys())
             return list(self._state.__dict__.keys())

    def __contains__(self, key):
        with self._lock:
            if isinstance(self._state, dict):
                return key in self._state
            return hasattr(self._state, key)

    def __len__(self):
        with self._lock:
            if isinstance(self._state, dict):
                return len(self._state)
            return len(self._state.__dict__)

    def __repr__(self):
        with self._lock:
            return f"<SharedStateManager proxying {type(self._state).__name__}>"
