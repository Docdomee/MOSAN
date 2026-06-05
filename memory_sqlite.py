# memory_sqlite.py
"""
SQLite-based vector memory implementation optimized for single-node multi-worker setup.
Provides drop-in replacement for VectorMemory with improved concurrency and performance.
"""

import os
import json
import sqlite3
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from state_manager import PERSISTENT_PATHS
from ui_logger import log

try:
    import tensorflow_text as text
except ImportError:
    text = None

# Process-level singleton — prevents reloading USE on every MemorySQLite instantiation
_USE_MODEL_SINGLETON = None

# Process-level singleton for RL MemoryRanker — same model_path_dir always yields same model
_RANKER_SINGLETON = None

# Disable JIT/XLA
tf.config.optimizer.set_jit(False)


class VectorMemorySQLite:
    """
    SQLite-based vector memory with WAL mode for concurrent access.
    Optimized for 4 concurrent Celery workers on single node.
    """
    
    def __init__(self):
        self.memory_dir = PERSISTENT_PATHS["long_term_memory_dir"]
        self.db_path = os.path.join(self.memory_dir, "memory.db")
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # Lazy loading for embedding model
        self.embedding_model = None
        self.embedding_dim = 512
        
        # Initialize database
        self._init_database()
        
        # --- NEW: Automated Migration from JSON ---
        self._migrate_from_json()
        
        # Initialize RL Ranker — reuse process-level singleton to avoid reloading Keras model
        global _RANKER_SINGLETON
        if _RANKER_SINGLETON is None:
            log("[Memory SQLite] Initializing RL MemoryRanker (first use)...")
            from reinforcement_learner import MemoryRanker
            _RANKER_SINGLETON = MemoryRanker(
                embedding_dim=self.embedding_dim,
                model_path_dir=PERSISTENT_PATHS["rl_models_dir"]
            )
            _RANKER_SINGLETON.load_weights()
        else:
            log("[Memory SQLite] Reusing existing RL MemoryRanker (singleton).")
        self.ranker = _RANKER_SINGLETON
        self.experience_buffer = []
        
        log(f"[Memory SQLite] Initialized with database: {self.db_path}")
    
    def _migrate_from_json(self):
        """Migrates legacy JSON memories to SQLite if the database is empty."""
        json_path = os.path.join(self.memory_dir, "global_agent_memory.json")
        
        if not os.path.exists(json_path):
            return
            
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if database is empty
            count = cursor.execute('SELECT COUNT(*) FROM memories').fetchone()[0]
            if count > 0:
                # log("[Memory SQLite] Database already has data. Skipping migration.")
                return
                
            log(f"[Memory SQLite] 📦 Database is empty. Migrating from {json_path}...")
            
            try:
                with open(json_path, "r") as f:
                    legacy_memories = json.load(f)
                
                if not legacy_memories:
                    log("[Memory SQLite] JSON memory file is empty.")
                    return
                
                log(f"[Memory SQLite] Found {len(legacy_memories)} memories in JSON. Bulk inserting...")
                
                # Prepare data for bulk insert
                to_insert = []
                for mem in legacy_memories:
                    finding_id = mem.get("finding_id")
                    if not finding_id:
                        import uuid
                        finding_id = str(uuid.uuid4())
                        
                    embedding = mem.get("embedding")
                    if embedding:
                        embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
                    else:
                        embedding_blob = np.zeros(self.embedding_dim, dtype=np.float32).tobytes()
                        
                    to_insert.append((
                        finding_id,
                        mem.get("finding", "No finding text"),
                        embedding_blob,
                        mem.get("source_experiment", "Unknown"),
                        mem.get("data_source_id", "Unknown"),
                        mem.get("data_type", "unknown"),
                        mem.get("data_info", "N/A")
                    ))
                
                cursor.executemany('''
                    INSERT OR IGNORE INTO memories (
                        finding_id, finding, embedding, source_experiment,
                        data_source_id, data_type, data_info
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', to_insert)
                
                conn.commit()
                log(f"[Memory SQLite] 🎉 Successfully migrated {len(to_insert)} memories from JSON to SQLite.")
                
            except Exception as e:
                log(f"[Memory SQLite ERROR] Migration failed: {e}")
                conn.rollback()
    
    def _init_database(self):
        """Initialize SQLite database with schema and optimizations."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Enable WAL mode for concurrent access
            cursor.execute('PRAGMA journal_mode=WAL')
            cursor.execute('PRAGMA synchronous=NORMAL')  # Fast + safe
            cursor.execute('PRAGMA cache_size=-64000')  # 64MB cache
            cursor.execute('PRAGMA temp_store=MEMORY')
            cursor.execute('PRAGMA mmap_size=268435456')  # 256MB mmap
            
            # Create main table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    finding_id TEXT UNIQUE NOT NULL,
                    finding TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    source_experiment TEXT NOT NULL,
                    data_source_id TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    data_info TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indices for fast queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_data_source_id ON memories(data_source_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_data_type ON memories(data_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_source_experiment ON memories(source_experiment)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at)')
            
            conn.commit()
            
            # Log stats
            count = cursor.execute('SELECT COUNT(*) FROM memories').fetchone()[0]
            log(f"[Memory SQLite] Database initialized. Current memories: {count}")
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0  # Wait up to 30s for lock
            )
            conn.row_factory = sqlite3.Row  # Access columns by name
            yield conn
        except sqlite3.Error as e:
            log(f"[Memory SQLite ERROR] Database error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate normalized embedding (same as JSON version)."""
        if self.embedding_model is None:
            if getattr(self, "_use_fallback", False):
                return np.zeros(self.embedding_dim, dtype=np.float32)
            
            log("[Memory SQLite] Loading Universal Sentence Encoder (CPU)...")
            global _USE_MODEL_SINGLETON
            try:
                if _USE_MODEL_SINGLETON is None:
                    with tf.device('/CPU:0'):
                        _USE_MODEL_SINGLETON = hub.load(
                            "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
                        )
                    log("[Memory SQLite] Encoder loaded successfully.")
                else:
                    log("[Memory SQLite] Reusing existing encoder (singleton).")
                self.embedding_model = _USE_MODEL_SINGLETON
            except Exception as e:
                log(f"[Memory SQLite ERROR] Failed to load encoder: {e}")
                self._use_fallback = True
                return np.zeros(self.embedding_dim, dtype=np.float32)
        
        try:
            embedding = self.embedding_model([text]).numpy()[0]
            return embedding / np.linalg.norm(embedding)
        except Exception as e:
            log(f"[Memory SQLite ERROR] Embedding generation failed: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def add(
        self,
        finding: str,
        source_experiment: str,
        data_source_id: str,
        finding_id: str = None,
        data_type: str = "unknown",
        data_info: str = "N/A"
    ):
        """Add new memory to database."""
        embedding = self._get_embedding(finding)
        embedding_blob = embedding.tobytes()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    INSERT INTO memories (
                        finding_id, finding, embedding, source_experiment,
                        data_source_id, data_type, data_info
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    finding_id, finding, embedding_blob, source_experiment,
                    data_source_id, data_type, data_info
                ))
                conn.commit()
                log(f"[Memory SQLite] Added memory ID: {finding_id}, type: {data_type}")
            except sqlite3.IntegrityError:
                log(f"[Memory SQLite] Memory {finding_id} already exists, skipping.")
    
    def save(self):
        """
        No-op method for compatibility with state_manager.
        SQLite auto-commits on each transaction, so explicit save is not needed.
        """
        # SQLite with auto-commit mode doesn't need explicit save
        # This method exists only for interface compatibility with VectorMemory (JSON)
        pass
    
    def search(
        self,
        query: str,
        current_data_source_id: Optional[str] = None,
        current_data_type: Optional[str] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for relevant memories using prioritized hierarchical search."""
        ENABLE_RL_RANKER = True
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Priority pools
            pool_prio_1 = []  # Same data_source_id
            pool_prio_2 = []  # Same data_type
            pool_prio_3 = []  # Everything else
            
            log("[Memory SQLite] Starting prioritized search...")
            
            # Efficient SQL queries for each pool
            if current_data_source_id:
                cursor.execute(
                    'SELECT * FROM memories WHERE data_source_id = ?',
                    (current_data_source_id,)
                )
                pool_prio_1 = [dict(row) for row in cursor.fetchall()]
            
            if current_data_type:
                cursor.execute(
                    'SELECT * FROM memories WHERE data_type = ? AND data_source_id != ?',
                    (current_data_type, current_data_source_id or '')
                )
                pool_prio_2 = [dict(row) for row in cursor.fetchall()]
            
            # Everything else
            cursor.execute('''
                SELECT * FROM memories 
                WHERE data_type != ? AND data_source_id != ?
            ''', (current_data_type or '', current_data_source_id or ''))
            pool_prio_3 = [dict(row) for row in cursor.fetchall()]
            
            log(f"[Memory SQLite] Pools - P1:{len(pool_prio_1)}, P2:{len(pool_prio_2)}, P3:{len(pool_prio_3)}")
            
            candidate_pool = pool_prio_1 + pool_prio_2 + pool_prio_3
            
            if not candidate_pool:
                return []
            
            # Keyword filter (architecture-based)
            known_architectures = [
                "1D_CNN", "2D_SPECTROGRAM", "2D_GAF", "STATS_MLP",
                "3D_VIDEO", "3D_GAF_VIDEO", "3D_DYNAMIC_GAF", "2D_CWT_SCALOGRAM"
            ]
            keyword_filter = None
            for arch in known_architectures:
                if arch in query:
                    keyword_filter = arch
                    break
            
            if keyword_filter:
                log(f"[Memory SQLite] Applying keyword filter: {keyword_filter}")
                filtered = [
                    mem for mem in candidate_pool
                    if keyword_filter in mem['finding'] or keyword_filter in mem['source_experiment']
                ]
                if filtered:
                    candidate_pool = filtered
            
            # Semantic search
            query_embedding = self._get_embedding(query)
            
            # Convert blobs to numpy arrays
            candidate_embeddings = []
            for mem in candidate_pool:
                emb_array = np.frombuffer(mem['embedding'], dtype=np.float32)
                candidate_embeddings.append(emb_array)
                mem['embedding'] = emb_array.tolist()  # For compatibility
            
            candidate_embeddings = np.array(candidate_embeddings).astype('float32')
            
            # Compute similarities
            # C2: If USB encoder failed (fallback mode), all embeddings are zero-vectors.
            # Use TF-IDF term-hit scoring as a meaningful alternative ranking.
            if getattr(self, "_use_fallback", False):
                query_terms = set(query.lower().split())
                similarities = np.array([
                    sum(1 for t in query_terms if t in mem.get("finding", "").lower())
                    for mem in candidate_pool
                ], dtype=np.float32)
            else:
                similarities = np.dot(candidate_embeddings, query_embedding)
            top_indices = np.argsort(similarities)[::-1][:min(20, len(candidate_pool))]
            
            candidate_memories = [candidate_pool[i] for i in top_indices]
            
            if not ENABLE_RL_RANKER:
                return candidate_memories[:k]
            
            # RL Re-ranking
            candidate_embeddings_for_rl = np.array([mem['embedding'] for mem in candidate_memories])
            q_values = self.ranker.predict_value(query_embedding, candidate_embeddings_for_rl)
            
            scored_memories = list(zip(candidate_memories, q_values))
            scored_memories.sort(key=lambda x: x[1], reverse=True)
            
            top_k_memories = [mem for mem, score in scored_memories[:k]]
            
            log(f"[Memory SQLite] Returning {len(top_k_memories)} memories")
            return top_k_memories
    
    def add_experience(self, query_embedding, chosen_memory_embedding, reward):
        """Add RL experience (same as JSON version)."""
        self.experience_buffer.append((query_embedding, chosen_memory_embedding, reward))
        log(f"[RL SQLite] Experience added. Reward: {reward:.2f}")
    
    def train_ranker(self):
        """Train RL ranker (same as JSON version)."""
        if not self.experience_buffer:
            log("[RL SQLite] No experiences to train on.")
            return
        
        log(f"[RL SQLite] Training with {len(self.experience_buffer)} experiences...")
        loss = self.ranker.train_step(self.experience_buffer)
        log(f"[RL SQLite] Training complete. Loss: {loss:.4f}")
        
        self.ranker.save_weights()
        self.experience_buffer = []
    
    def get_memory_count(self) -> int:
        """Get total number of memories."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            count = cursor.execute('SELECT COUNT(*) FROM memories').fetchone()[0]
            return count
