# memory_utils.py
# CRITICAL: Set TensorFlow environment variables BEFORE import to prevent JIT issues
import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TF logging

# Now safe to import
import json
import tempfile
from typing import Optional

# File locking imports (cross-platform)
try:
    import fcntl  # Unix/Linux
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False
    
try:
    import msvcrt  # Windows
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False

import faiss
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Process-level singleton — prevents reloading USE on every VectorMemory instantiation
_USE_MODEL_SINGLETON = None

# Process-level singleton for RL MemoryRanker
_RANKER_SINGLETON = None
try:
    import tensorflow_text as text
except ImportError:
    print("[Memory WARNING] tensorflow_text not found. Universal Sentence Encoder might fail if it relies on custom ops.")
    text = None
from reinforcement_learner import MemoryRanker
from state_manager import PERSISTENT_PATHS
from ui_logger import log

# Disable JIT/XLA to prevent "Graph execution error" with Universal Sentence Encoder on some clusters
tf.config.optimizer.set_jit(False)


class VectorMemory:
    """
    Manages the agent's long-term vector memory using a unified, global storage.
    This architecture is enhanced with a Reinforcement Learning-based re-ranking system.
    """

    def __init__(self):
        # Usa il percorso centralizzato per la memoria a lungo termine
        self.memory_dir = PERSISTENT_PATHS["long_term_memory_dir"]
        self.memory_file = os.path.join(self.memory_dir, "global_agent_memory.json")
        os.makedirs(self.memory_dir, exist_ok=True)

        # LAZY LOADING: Don't load encoder until first use (prevents cluster GPU issues)
        self.embedding_model = None
        self.embedding_dim = 512

        global _RANKER_SINGLETON
        if _RANKER_SINGLETON is None:
            log("[Memory] Initializing global RL MemoryRanker (first use)...")
            _RANKER_SINGLETON = MemoryRanker(embedding_dim=self.embedding_dim, model_path_dir=PERSISTENT_PATHS["rl_models_dir"])
            _RANKER_SINGLETON.load_weights()
        else:
            log("[Memory] Reusing existing RL MemoryRanker (singleton).")
        self.ranker = _RANKER_SINGLETON
        self.experience_buffer = []

        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.memory = []
        self._load()

    def _load(self):
        """Loads existing memories from the global JSON file with file locking."""
        if not os.path.exists(self.memory_file):
            log("[Memory] No existing memory file found. Starting fresh.")
            return
            
        lock_file_path = self.memory_file + '.lock'
        
        try:
            # Acquire shared lock for reading
            with open(lock_file_path, 'w') as lock_file:
                if HAS_FCNTL:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)  # Shared lock
                    log("[Memory] Acquired read lock (fcntl)")
                elif HAS_MSVCRT:
                    # Windows doesn't have shared locks in the same way, use exclusive briefly
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
                    log("[Memory] Acquired read lock (msvcrt)")
                
                try:
                    with open(self.memory_file, "r") as f:
                        self.memory = json.load(f)
                    
                    if self.memory:
                        embeddings = np.array([item["embedding"] for item in self.memory]).astype("float32")
                        self.index.add(embeddings)
                        log(f"[Memory] Loaded {len(self.memory)} memories from global store: {self.memory_file}")
                    else:
                        log("[Memory] Memory file is empty")
                        
                except (json.JSONDecodeError, KeyError) as e:
                    log(f"[Memory ERROR] Could not load memory file {self.memory_file}. A new file will be created. Error: {e}")
                    self.memory = []
                    
                finally:
                    # Release lock
                    if HAS_FCNTL:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                    elif HAS_MSVCRT:
                        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                        
        except Exception as e:
            log(f"[Memory] Error during load (proceeding with empty memory): {e}")
            self.memory = []

    def save(self):
        """
        Saves the memory list to the global file with file locking to prevent race conditions.
        Uses atomic write (temp file + rename) for additional safety.
        """
        log(f"[Memory] Saving {len(self.memory)} memories to {self.memory_file}...")
        
        # Create a temporary file in the same directory to ensure atomic write
        temp_fd, temp_path = tempfile.mkstemp(
            dir=self.memory_dir,
            prefix='.tmp_memory_',
            suffix='.json'
        )
        
        try:
            # Write to temp file first
            with os.fdopen(temp_fd, 'w') as temp_file:
                json.dump(self.memory, temp_file, indent=4)
            
            # Now acquire lock and atomically rename
            # Open the target file for locking (create if doesn't exist)
            lock_file_path = self.memory_file + '.lock'
            with open(lock_file_path, 'w') as lock_file:
                # Acquire exclusive lock
                if HAS_FCNTL:
                    # Unix/Linux locking
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                    log("[Memory] Acquired file lock (fcntl)")
                elif HAS_MSVCRT:
                    # Windows locking
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
                    log("[Memory] Acquired file lock (msvcrt)")
                else:
                    log("[Memory] WARNING: No file locking available on this platform!")
                
                try:
                    # Atomic rename (overwrites target file)
                    # On Windows, need to remove target first if it exists
                    if os.path.exists(self.memory_file):
                        if os.name == 'nt':  # Windows
                            os.replace(temp_path, self.memory_file)
                        else:  # Unix/Linux
                            os.rename(temp_path, self.memory_file)
                    else:
                        os.rename(temp_path, self.memory_file)
                    
                    log(f"[Memory] Successfully saved {len(self.memory)} memories atomically")
                    
                finally:
                    # Release lock
                    if HAS_FCNTL:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                        log("[Memory] Released file lock (fcntl)")
                    elif HAS_MSVCRT:
                        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                        log("[Memory] Released file lock (msvcrt)")
                        
        except Exception as e:
            log(f"[Memory] CRITICAL ERROR during save: {e}")
            # Clean up temp file if it still exists
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            raise

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generates a normalized embedding for a given text."""
        # LAZY LOADING: Load encoder on first use
        if self.embedding_model is None:
            # Check if we already tried and failed
            if getattr(self, "_use_fallback", False):
                return np.zeros(self.embedding_dim, dtype=np.float32)

            log("[Memory] Loading Universal Sentence Encoder (first use - forcing CPU for stability)...")
            global _USE_MODEL_SINGLETON
            try:
                if _USE_MODEL_SINGLETON is None:
                    with tf.device('/CPU:0'):
                        _USE_MODEL_SINGLETON = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
                    log("[Memory] Universal Sentence Encoder loaded successfully on CPU.")
                else:
                    log("[Memory] Reusing existing encoder (singleton).")
                self.embedding_model = _USE_MODEL_SINGLETON
            except Exception as e:
                log(f"[Memory ERROR] Failed to load Universal Sentence Encoder: {e}")
                log("[Memory] SWITCHING TO FALLBACK MODE: Semantic search will be disabled, relying on keyword matching.")
                self._use_fallback = True
                return np.zeros(self.embedding_dim, dtype=np.float32)
        
        try:
            embedding = self.embedding_model([text]).numpy()[0]
            return embedding / np.linalg.norm(embedding)
        except Exception as e:
             log(f"[Memory ERROR] Error generating embedding: {e}")
             return np.zeros(self.embedding_dim, dtype=np.float32)

    # 1. Add data_type and data_info to the function definition
    def add(
        self,
        finding: str,
        source_experiment: str,
        data_source_id: str,
        finding_id: str = None,
        data_type: str = "unknown",
        data_info: str = "N/A",
    ):
        """
        Adds a new finding to memory, tagged with its source experiment and a unique ID.
        """
        embedding = self._get_embedding(finding)
        self.index.add(np.array([embedding]).astype("float32"))

        memory_entry = {
            "finding_id": finding_id,
            "finding": finding,
            "embedding": embedding.tolist(),
            "source_experiment": source_experiment,
            "data_source_id": data_source_id,
            # 2. These variables are now correctly defined from the arguments
            "data_type": data_type,
            "data_info": data_info,
        }
        self.memory.append(memory_entry)

        self.save()
        # 3. This log message will now work correctly
        log(f"[Memory] New memory from '{source_experiment}' on data '{data_type}' added with ID: {finding_id}")

    def add_experience(self, query_embedding, chosen_memory_embedding, reward):
        """Adds an experience tuple to the buffer for future training."""
        self.experience_buffer.append((query_embedding, chosen_memory_embedding, reward))
        log(f"[RL] Experience added to buffer. Reward: {reward:.2f}")

    def train_ranker(self):
        """Trains the MemoryRanker using the collected experiences and then clears the buffer."""
        if not self.experience_buffer:
            log("[RL] No new experiences in buffer, skipping training.")
            return

        log(f"[RL] Training the global MemoryRanker with {len(self.experience_buffer)} new experiences...")
        loss = self.ranker.train_step(self.experience_buffer)
        log(f"[RL] Training step completed. Loss: {loss:.4f}")

        self.ranker.save_weights()
        self.experience_buffer = []
        log("[RL] Experience buffer cleared.")

    def search(
        self,
        query: str,
        current_data_source_id: Optional[str] = None,
        current_data_type: Optional[str] = None,
        k: int = 5,
    ) -> list[dict]:
        """
        Searches for the most RELEVANT memories using a hierarchical, prioritized process.
        """
        ENABLE_RL_RANKER = True

        if not self.memory:
            return []

        # --- NUOVA LOGICA DI RICERCA PRIORITARIA ---

        pool_prio_1 = []  # Corrispondenza esatta (stesso ID)
        pool_prio_2 = []  # Corrispondenza di tipo (stesso data_type)
        pool_prio_3 = []  # Tutto il resto

        log("[Memory] Starting prioritized memory search...")
        for mem in self.memory:
            mem_source_id = mem.get("data_source_id")
            mem_data_type = mem.get("data_type")

            if current_data_source_id and mem_source_id == current_data_source_id:
                pool_prio_1.append(mem)
            elif current_data_type and mem_data_type == current_data_type:
                pool_prio_2.append(mem)
            else:
                pool_prio_3.append(mem)

        # Uniamo i pool in ordine di priorità. I ricordi più importanti sono all'inizio.
        candidate_pool = pool_prio_1 + pool_prio_2 + pool_prio_3

        log(
            f"[Memory] Pool sizes - Priority 1: {len(pool_prio_1)}, Priority 2: {len(pool_prio_2)}, Priority 3: {len(pool_prio_3)}"
        )

        # --- FINE DELLA NUOVA LOGICA ---

        if not candidate_pool:
            return []

        # Il resto della funzione (Filtro per Keyword, Ricerca Semantica, Riordino RL)
        # ora opera sul 'candidate_pool' che è già intelligentemente ordinato per rilevanza.

        # --- START OF HYBRID SEARCH LOGIC ---
        known_architectures = [
            "1D_CNN",
            "2D_SPECTROGRAM",
            "2D_GAF",
            "STATS_MLP",
            "3D_VIDEO",
            "3D_GAF_VIDEO",
            "2D_CWT_SCALOGRAM",
        ]
        keyword_filter = None
        for arch in known_architectures:
            if arch in query:
                keyword_filter = arch
                break

        if keyword_filter:
            log(f"[Memory] Hybrid Search: Applying keyword filter for '{keyword_filter}'")
            filtered_by_keyword = [
                mem
                for mem in candidate_pool
                if keyword_filter in mem.get("finding", "") or keyword_filter in mem.get("source_experiment", "")
            ]
            if filtered_by_keyword:
                candidate_pool = filtered_by_keyword
            else:
                log(
                    f"[Memory] No memories found matching keyword '{keyword_filter}'. Falling back to prioritized pool."
                )

        # --- END OF HYBRID SEARCH LOGIC ---

        if not candidate_pool:
            return []

        query_embedding = self._get_embedding(query)

        # 3. Semantic Search on the (now much smaller and more relevant) candidate pool
        candidate_embeddings = np.array([mem["embedding"] for mem in candidate_pool]).astype("float32")
        temp_index = faiss.IndexFlatIP(self.embedding_dim)
        temp_index.add(candidate_embeddings)

        num_candidates = min(len(candidate_pool), 20)
        distances, indices = temp_index.search(np.array([query_embedding]).astype("float32"), num_candidates)

        candidate_indices = [i for i in indices[0] if i != -1]
        if not candidate_indices:
            return []

        candidate_memories = [candidate_pool[i] for i in candidate_indices]

        if not ENABLE_RL_RANKER:
            log("[Memory] RL Ranker is DISABLED. Returning top k results by similarity from the filtered pool.")
            return candidate_memories[:k]

        # Stage 2: Re-ranking (RL)
        candidate_embeddings_for_rl = np.array([mem["embedding"] for mem in candidate_memories])
        log(f"[RL] Re-ranking {len(candidate_memories)} candidate memories...")
        q_values = self.ranker.predict_value(query_embedding, candidate_embeddings_for_rl)

        scored_memories = list(zip(candidate_memories, q_values))
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        top_k_memories = [mem for mem, score in scored_memories[:k]]

        log(f"[Memory] Search complete. Returning the {len(top_k_memories)} most promising memories.")
        return top_k_memories


# ============================================================================
# FACTORY FUNCTION FOR BACKEND SELECTION
# ============================================================================

def create_vector_memory(backend: str = "json", fallback_to_json: bool = True):
    """
    Factory function to create VectorMemory with configurable backend.
    
    Args:
        backend: "json" or "sqlite"
        fallback_to_json: If True and SQLite fails, fallback to JSON backend
    
    Returns:
        VectorMemory instance (JSON or SQLite)
    """
    # Check environment variable override
    backend = os.environ.get("VECTOR_MEMORY_BACKEND", backend).lower()
    
    if backend == "sqlite":
        try:
            log("[Memory Factory] Attempting to load SQLite backend...")
            from memory_sqlite import VectorMemorySQLite
            memory = VectorMemorySQLite()
            log("[Memory Factory] ✅ SQLite backend loaded successfully")
            return memory
        except Exception as e:
            log(f"[Memory Factory] ❌ SQLite backend failed: {e}")
            if fallback_to_json:
                log("[Memory Factory] ⚠️  Falling back to JSON backend...")
                return VectorMemory()
            else:
                raise
    elif backend == "json":
        log("[Memory Factory] Using JSON backend")
        return VectorMemory()
    else:
        log(f"[Memory Factory] Unknown backend '{backend}', defaulting to JSON")
        return VectorMemory()
