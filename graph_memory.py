# graph_memory.py

import os
import uuid
import json
import shutil
import time
import threading
from collections import deque
from copy import deepcopy
from typing import List, Dict, Any, Tuple, Optional

import networkx as nx
import numpy as np

from state_manager import PERSISTENT_PATHS
from ui_logger import log

# MOSAN Constants
SIMILARITY_THRESHOLD = 0.05  # Normalized Euclidean Distance threshold for merging
MAX_NODES_FOR_BRUTE_FORCE = 1000 # Beyond this, we might need a KDTree

# Q-vector schema V2.1: [d_acc, d_overfit, d_speed, d_latency, d_memory]
# Centralized indices so saturation detectors and metric specialists agree on layout.
Q_VEC_ACC = 0
Q_VEC_OVERFIT = 1
Q_VEC_SPEED = 2
Q_VEC_LATENCY = 3
Q_VEC_MEMORY = 4

# Saturation detector internal constants
MIN_NODES_FOR_PERCENTILE = 10
RECENT_EDGES_LOG_MAXLEN = 100

class StrategicGraphMemory:
    """
    MOSAN (Multi-Objective State-Action Network) V2.1
    Organic Growth Memory:
    - Nodes are 'Anchors' in vector space (not bins).
    - Edges store raw Impact Vectors (not pre-calculated rewards).
    - Decisions are made by applying Contextual Lenses at runtime.
    """

    def __init__(self, dataset_id: str = "global", ghost_threshold: int = 10):
        """
        Initializes the Organic Strategic Memory.
        Supports dataset-isolated graph files (Ghost Node / Ephemeral Overlay architecture).

        Args:
            dataset_id: Identifier for the active dataset. Defaults to "global" for backward compatibility.
            ghost_threshold: Inject ghost nodes from similar datasets only when own graph has fewer than
                             this many nodes (cold-start guard). Set to 0 to disable injection entirely.
        """
        self.dataset_id = dataset_id
        self.ghost_threshold = ghost_threshold

        # Reentrant lock serializing graph mutation + serialization. The Decider runs
        # GPU tools in parallel threads, each calling add_organic_experience (which also
        # triggers save()). Without this, one thread deep-copies the graph in save()
        # while another adds a node/edge -> "dictionary changed size during iteration".
        self._lock = threading.RLock()

        self.memory_dir = PERSISTENT_PATHS.get(
            "strategic_memory_dir", os.path.join(PERSISTENT_PATHS["processed_data_dir"], "strategic_memory")
        )

        # Per-dataset subdirectory — created here, not by state_manager
        self.dataset_dir = os.path.join(self.memory_dir, dataset_id)
        os.makedirs(self.dataset_dir, exist_ok=True)

        self.graph_file = os.path.join(self.dataset_dir, "strategic_graph_organic.graphml")

        # Legacy migration: flat root-level file from before Ghost Node architecture
        self.legacy_graph_file = os.path.join(self.memory_dir, "strategic_graph.graphml")
        legacy_organic = os.path.join(self.memory_dir, "strategic_graph_organic.graphml")

        # If a root-level organic file exists and the per-dataset one doesn't, migrate it into global/
        if dataset_id == "global":
            if os.path.exists(legacy_organic) and not os.path.exists(self.graph_file):
                try:
                    shutil.move(legacy_organic, self.graph_file)
                    log(f"[MOSAN] Migrated root-level organic graph → {self.graph_file}")
                except Exception as e:
                    log(f"[MOSAN] Warning: Migration failed: {e}")

        # Original legacy (MPD-style) archival
        if os.path.exists(self.legacy_graph_file) and not os.path.exists(self.graph_file):
            self._archive_legacy_graph()

        self.graph = self.load()

        # In-memory log of recent edge updates for saturation detectors (Hawk slope, Sage variance).
        # Not persisted: saturation operates on current-session signal only.
        self._recent_edges_log: deque = deque(maxlen=RECENT_EDGES_LOG_MAXLEN)

        # Ghost Node injection for cold-start
        if ghost_threshold > 0:
            self._inject_ephemeral_wisdom(ghost_threshold)

    def save_mosan_metadata(self, representation_type: str, data_domain: str = "", num_classes: Optional[int] = None):
        """
        Creates or updates metadata.json for this dataset in its strategic_memory folder.
        - If the file doesn't exist, it is created (idempotent on subsequent calls).
        - num_classes is intentionally optional: if not provided, the file is written without it
          and the user receives a log message with the path to fill it in manually.
        - Existing keys are preserved; only missing keys are added.
        """
        meta_path = os.path.join(self.dataset_dir, "metadata.json")
        existing = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    existing = json.load(f)
            except Exception as e:
                log(f"[MOSAN] Warning: Could not read existing metadata, will overwrite. Error: {e}")

        updated = False
        if "representation_type" not in existing:
            existing["representation_type"] = representation_type
            updated = True
        if data_domain and "data_domain" not in existing:
            existing["data_domain"] = data_domain
            updated = True
        if num_classes is not None and "num_classes" not in existing:
            existing["num_classes"] = num_classes
            updated = True

        if updated:
            try:
                with open(meta_path, "w") as f:
                    json.dump(existing, f, indent=4)
                log(f"[MOSAN] Saved metadata for '{self.dataset_id}': {existing}")
            except Exception as e:
                log(f"[MOSAN] Error: Could not write metadata.json: {e}")
                return

        if existing.get("num_classes") is None:
            log(
                f"[MOSAN] ⚠ 'num_classes' not set for '{self.dataset_id}' — ghost injection disabled until you add it.\n"
                f"        → Edit: {meta_path}"
            )

    def _archive_legacy_graph(self):
        """Archives the old MPD-style graph to avoid confusion."""
        log("[MOSAN] Detected legacy graph. Archiving...")
        archive_path = self.legacy_graph_file + ".bak_pre_mosan"
        try:
            shutil.move(self.legacy_graph_file, archive_path)
            log(f"[MOSAN] Archived legacy graph to {archive_path}")
        except Exception as e:
            log(f"[MOSAN] Warning: Failed to archive legacy graph: {e}")

    def _inject_ephemeral_wisdom(self, ghost_threshold: int):
        """
        Ghost Node injection: injects high-Q edges from similar neighbor datasets into the
        active graph during cold start. Injected nodes are tagged ephemeral=True and are
        never persisted by save().

        Similarity criterion: same representation_type AND |num_classes_A - num_classes_B| < 5.
        Limited to the 3 most recently modified neighbor datasets (latency guard).
        """
        node_count = len(self.graph.nodes)

        if node_count >= ghost_threshold:
            log(f"[MOSAN] Graph has {node_count} nodes — skipping ghost injection (above threshold).")
            return

        # Load own metadata for similarity comparison
        own_meta_path = os.path.join(self.dataset_dir, "metadata.json")
        if not os.path.exists(own_meta_path):
            log(
                f"[MOSAN] No metadata.json for '{self.dataset_id}' — skipping ghost injection.\n"
                f"        Ghost cross-dataset transfer is disabled. Call save_mosan_metadata() to enable it."
            )
            return

        try:
            with open(own_meta_path) as f:
                own_meta = json.load(f)
        except Exception as e:
            log(f"[MOSAN] Failed to read own metadata: {e}")
            return

        own_repr = own_meta.get("representation_type", "")
        own_classes = own_meta.get("num_classes", None)
        own_domain = own_meta.get("data_domain", "")

        if not own_repr:
            log(
                f"[MOSAN] metadata.json for '{self.dataset_id}' is missing 'representation_type' — skipping ghost injection.\n"
                f"        → Edit: {own_meta_path}"
            )
            return

        if own_classes is None:
            log(
                f"[MOSAN] metadata.json for '{self.dataset_id}' is missing 'num_classes'.\n"
                f"        Ghost injection will be skipped. Add it manually to enable cross-dataset transfer.\n"
                f"        → Edit: {own_meta_path}"
            )
            return

        # Scan sibling directories (exclude own and global/)
        try:
            siblings = [
                d for d in os.listdir(self.memory_dir)
                if os.path.isdir(os.path.join(self.memory_dir, d))
                and d != self.dataset_id
                and d != "global"
            ]
        except Exception as e:
            log(f"[MOSAN] Cannot scan memory_dir: {e}")
            return

        if not siblings:
            log(f"[MOSAN] No neighbor datasets found for ghost injection.")
            return

        # Sort siblings by most recently modified (latency guard: only top 3)
        def _mtime(name):
            try:
                return os.path.getmtime(os.path.join(self.memory_dir, name))
            except Exception:
                return 0

        siblings.sort(key=_mtime, reverse=True)
        siblings = siblings[:3]

        injected_nodes = 0
        injected_from = 0

        for sibling in siblings:
            sibling_dir = os.path.join(self.memory_dir, sibling)
            meta_path = os.path.join(sibling_dir, "metadata.json")
            graph_path = os.path.join(sibling_dir, "strategic_graph_organic.graphml")

            if not os.path.exists(meta_path) or not os.path.exists(graph_path):
                continue

            try:
                with open(meta_path) as f:
                    nb_meta = json.load(f)
            except Exception:
                continue

            nb_repr = nb_meta.get("representation_type", "")
            nb_classes = nb_meta.get("num_classes", None)
            nb_domain = nb_meta.get("data_domain", "")

            # Similarity check: same representation_type is mandatory
            if nb_repr != own_repr:
                continue
            # Affinity: same data_domain (e.g. SERS_spectral) counts as similar regardless of class count;
            # otherwise fall back to strict class-count proximity (|diff| < 5)
            same_domain = own_domain and nb_domain and own_domain == nb_domain
            if not same_domain:
                if nb_classes is None or abs(nb_classes - own_classes) >= 5:
                    continue

            # Load neighbor graph
            try:
                nb_graph = nx.read_graphml(graph_path)
                nb_graph = nx.MultiDiGraph(nb_graph)
            except Exception as e:
                log(f"[MOSAN] Failed to load neighbor graph '{sibling}': {e}")
                continue

            # Deserialize q_vector for sorting
            for _, _, k, data in nb_graph.edges(data=True, keys=True):
                if "q_vector" in data and isinstance(data["q_vector"], str):
                    try:
                        data["q_vector"] = json.loads(data["q_vector"])
                    except Exception:
                        data["q_vector"] = []

            # Find top 5 nodes by outgoing Q-value (first element)
            node_q_scores = []
            for node in nb_graph.nodes():
                best_q = float("-inf")
                for _, _, data in nb_graph.edges(node, data=True):
                    q_vec = data.get("q_vector", data.get("impact_vector", []))
                    if q_vec and len(q_vec) > 0 and q_vec[0] > best_q:
                        best_q = q_vec[0]
                if best_q != float("-inf"):
                    node_q_scores.append((node, best_q))

            node_q_scores.sort(key=lambda x: x[1], reverse=True)
            top_nodes = [n for n, _ in node_q_scores[:5]]

            # Build ghost_id map for this sibling so we can inject edges safely
            ghost_id_map = {node_id: f"ghost_{sibling}_{node_id}" for node_id in top_nodes}

            for node_id in top_nodes:
                ghost_id = ghost_id_map[node_id]
                node_attrs = dict(nb_graph.nodes[node_id])
                # Deserialize vector if serialized
                if "vector" in node_attrs and isinstance(node_attrs["vector"], str):
                    try:
                        node_attrs["vector"] = json.loads(node_attrs["vector"])
                    except Exception:
                        pass
                node_attrs["ephemeral"] = True
                node_attrs["ghost_source"] = sibling
                self.graph.add_node(ghost_id, **node_attrs)
                injected_nodes += 1

            # D1: Inject top-3 outgoing edges for each ghost node.
            # Only inject edges whose target is also a ghost node (no dangling edges).
            injected_edges = 0
            for node_id, ghost_id in ghost_id_map.items():
                out_edges = sorted(
                    nb_graph.edges(node_id, data=True, keys=True),
                    key=lambda e: (e[3].get("q_vector") or [0])[0] if e[3].get("q_vector") else 0,
                    reverse=True,
                )[:3]
                for u, v, k, edata in out_edges:
                    ghost_target = ghost_id_map.get(v)
                    if ghost_target and ghost_target in self.graph.nodes:
                        edge_attrs = dict(edata)
                        # Deserialize q_vector if serialized
                        if "q_vector" in edge_attrs and isinstance(edge_attrs["q_vector"], str):
                            try:
                                edge_attrs["q_vector"] = json.loads(edge_attrs["q_vector"])
                            except Exception:
                                pass
                        edge_attrs["ephemeral"] = True
                        edge_attrs["ghost_source"] = sibling
                        self.graph.add_edge(ghost_id, ghost_target, **edge_attrs)
                        injected_edges += 1

            if top_nodes:
                injected_from += 1
                log(f"[MOSAN] Injected {len(top_nodes)} ghost nodes + {injected_edges} ghost edges from '{sibling}'.")

        if injected_nodes > 0:
            log(f"[MOSAN] Injected {injected_nodes} Ghost Nodes from {injected_from} neighbor datasets for cold-start optimization.")
        else:
            log(f"[MOSAN] Ghost injection found no qualifying neighbors for '{self.dataset_id}'.")

    def save(self):
        """Saves the graph to disk, handling vector serialization."""
        if not self.graph or not self.graph.nodes:
            log("[MOSAN] Graph is empty, skipping save.")
            return
        try:
            # Atomic snapshot: block concurrent mutations (add_organic_experience) while
            # we deep-copy, else the graph dict changes size during iteration.
            with self._lock:
                graph_to_save = deepcopy(self.graph)

            # Ephemeral guard — nodes: identify ghost nodes first
            ephemeral_nodes = {n for n, d in graph_to_save.nodes(data=True) if d.get("ephemeral")}
            skipped_nodes = len(ephemeral_nodes)

            # Ephemeral guard — edges: count edges touching any ephemeral node BEFORE removal (double-guard)
            skipped_edges = sum(
                1 for u, v in graph_to_save.edges()
                if u in ephemeral_nodes or v in ephemeral_nodes
            )

            # Remove ephemeral nodes (NetworkX auto-removes their incident edges)
            for n in ephemeral_nodes:
                graph_to_save.remove_node(n)

            # Independent edge filter as safety belt: any edge that slipped through (shouldn't happen, but guards against bugs)
            edges_to_remove = [
                (u, v, k) for u, v, k in graph_to_save.edges(keys=True)
                if u in ephemeral_nodes or v in ephemeral_nodes
            ]
            for u, v, k in edges_to_remove:
                graph_to_save.remove_edge(u, v, k)

            if skipped_nodes > 0:
                log(f"[MOSAN] Skipped {skipped_nodes} ephemeral nodes and {skipped_edges} ephemeral edges during save.")

            # Serialize Vector Attributes for GraphML (which implies simple types)
            for node, data in graph_to_save.nodes(data=True):
                if "vector" in data and isinstance(data["vector"], (list, np.ndarray)):
                    data["vector"] = json.dumps(data["vector"] if isinstance(data["vector"], list) else data["vector"].tolist())

            for u, v, data in graph_to_save.edges(data=True):
                if "impact_vector" in data and isinstance(data["impact_vector"], (list, np.ndarray)):
                    data["impact_vector"] = json.dumps(data["impact_vector"] if isinstance(data["impact_vector"], list) else data["impact_vector"].tolist())
                if "raw_rewards" in data:
                     # Flatten or summarize raw rewards if too large, strictly string for GraphML
                     data["raw_rewards"] = json.dumps(data["raw_rewards"])
                if "full_arguments" in data and isinstance(data["full_arguments"], dict):
                    # Use default=str to safely serialize any remaining odd objects (like functions)
                    data["full_arguments"] = json.dumps(data["full_arguments"], default=str)
                if "q_vector" in data and isinstance(data["q_vector"], (list, np.ndarray)):
                    data["q_vector"] = json.dumps(data["q_vector"] if isinstance(data["q_vector"], list) else data["q_vector"].tolist(), default=str)

            # Docker-Safe Save Strategy: Use local temp directory first
            import tempfile
            
            # Ensure destination directory exists
            os.makedirs(self.dataset_dir, exist_ok=True)
            
            # Write to system temp first (avoids Docker mount I/O issues)
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.graphml') as tmp:
                local_temp_path = tmp.name
            
            try:
                # Write graph to local filesystem temp
                nx.write_graphml(graph_to_save, local_temp_path)
                
                if not os.path.exists(local_temp_path):
                    log(f"[MOSAN ERROR] Failed to create local temp file {local_temp_path}")
                    return
                
                # Move from local temp to Docker volume using robust strategy
                success = False
                
                try:
                    shutil.move(local_temp_path, self.graph_file)
                    success = True
                except Exception as e1:
                    # Fallback: manual copy + delete
                    try:
                        shutil.copy2(local_temp_path, self.graph_file)
                        os.remove(local_temp_path)
                        success = True
                    except Exception as e2:
                        log(f"[MOSAN ERROR] All save strategies failed. Last: {e2}")
                        # Cleanup
                        try:
                            os.remove(local_temp_path)
                        except:
                            pass
                
                if success:
                    pass  # log("[MOSAN] Graph saved successfully.")
                    
            except Exception as write_error:
                log(f"[MOSAN ERROR] Failed to write GraphML: {write_error}")
                if os.path.exists(local_temp_path):
                    try:
                        os.remove(local_temp_path)
                    except:
                        pass

            # log("[MOSAN] Organic Graph saved successfully.")
        except Exception as e:
            import traceback
            log(f"[MOSAN ERROR] Could not save graph: {e}")
            log(f"[MOSAN TRACE] {traceback.format_exc()}")

    def load(self):
        """Loads the graph and deserializes vectors."""
        if os.path.exists(self.graph_file):
            try:
                log(f"[MOSAN] Loading Organic Graph from {self.graph_file}")
                loaded_graph = nx.read_graphml(self.graph_file)
                graph = nx.MultiDiGraph(loaded_graph)

                # Deserialize Vectors
                for node, data in graph.nodes(data=True):
                    if "vector" in data and isinstance(data["vector"], str):
                        data["vector"] = json.loads(data["vector"])
                
                for u, v, k, data in graph.edges(data=True, keys=True):
                    if "impact_vector" in data and isinstance(data["impact_vector"], str):
                        data["impact_vector"] = json.loads(data["impact_vector"])
                    if "raw_rewards" in data and isinstance(data["raw_rewards"], str):
                        try:
                            data["raw_rewards"] = json.loads(data["raw_rewards"])
                        except:
                            data["raw_rewards"] = []
                    if "full_arguments" in data and isinstance(data["full_arguments"], str):
                        try:
                            data["full_arguments"] = json.loads(data["full_arguments"])
                        except:
                            data["full_arguments"] = {}
                    if "q_vector" in data and isinstance(data["q_vector"], str):
                        try:
                            data["q_vector"] = json.loads(data["q_vector"])
                        except:
                            data["q_vector"] = []

                log(f"[MOSAN] Loaded dataset-specific graph for '{self.dataset_id}': {len(graph.nodes)} nodes, {len(graph.edges)} edges.")
                return graph
            except Exception as e:
                log(f"[MOSAN ERROR] Corrupt graph file. Starting fresh. Error: {e}")
                return nx.MultiDiGraph()
        else:
            log(f"[MOSAN] No graph found for '{self.dataset_id}'. Creating Tabula Rasa.")
            return nx.MultiDiGraph()

    def _get_or_create_anchor(self, state_vector: List[float], architecture: str) -> str:
        """
        ORGANIC GROWTH CORE:
        Finds the nearest 'Anchor' node. 
        If distance < Threshold -> Return Link.
        Else -> Spawn New Node.
        """
        # Ensure flat list
        vec = np.array(state_vector).flatten()
        
        # 1. Search existing nodes (Brute force is fine for <1000 nodes)
        best_node = None
        min_dist = float('inf')
        
        for node, data in self.graph.nodes(data=True):
            # Optimization: Only match same architecture family nodes?
            # User wants extensive network. Let's strictly match architecture type for now to avoid confusion
            # e.g. a 1D state shouldn't map to a 2D state anchor.
            stored_arch = data.get("architecture", "unknown")
            if stored_arch != architecture:
                continue
                
            stored_vec = np.array(data["vector"])
            
            # STATE SPACE EVOLUTION FIX: 
            # If dimensions differ (e.g. tools/models added), pad with zeros for comparison
            # This ensures organic growth isn't lost when the state size evolves.
            if vec.shape != stored_vec.shape:
                max_len = max(len(vec), len(stored_vec))
                v1 = np.pad(vec, (0, max_len - len(vec)))
                v2 = np.pad(stored_vec, (0, max_len - len(stored_vec)))
                dist = np.linalg.norm(v1 - v2)
            else:
                dist = np.linalg.norm(vec - stored_vec)
            
            if dist < min_dist:
                min_dist = dist
                best_node = node

        # 2. Threshold Check
        if best_node and min_dist < SIMILARITY_THRESHOLD:
            # log(f"[MOSAN] Matched existing anchor '{best_node}' (Dist: {min_dist:.4f})")
            return best_node
        
        # 3. Spawn New Node
        new_id = f"{architecture}_{uuid.uuid4().hex[:8]}"
        self.graph.add_node(
            new_id, 
            vector=vec.tolist(), 
            architecture=architecture,
            creation_step=len(self.graph.nodes)
        )
        log(f"[MOSAN] Spawning NEW Anchor: '{new_id}'")
        return new_id

    def add_organic_experience(self, *args, **kwargs):
        """Thread-safe entry point for recording a transition.

        Serializes the whole mutation (anchor resolution, node/edge writes) and the
        save() it triggers, so parallel GPU tools cannot mutate the graph while another
        thread is deep-copying it in save(). RLock makes the nested save() call reentrant.
        """
        with self._lock:
            return self._add_organic_experience_impl(*args, **kwargs)

    def _add_organic_experience_impl(
        self,
        state_before: List[float],
        state_after: List[float],
        action_name: str,
        impact_vector: List[float], # [d_acc, d_stab, d_time, d_nov]
        current_architecture: str,
        finding_id: str = None,
        full_arguments: dict = None
    ):
        """
        Records a transition in the organic graph.
        """
        # 1. Resolve Anchors
        anchor_from = self._get_or_create_anchor(state_before, current_architecture)
        anchor_to = self._get_or_create_anchor(state_after, current_architecture) # Usually same arch, unless mutation tool

        # 2. Create/Update Edge
        if self.graph.has_edge(anchor_from, anchor_to, key=action_name):
            edge_data = self.graph.edges[anchor_from, anchor_to, action_name]
            edge_data["visits"] += 1
            
            # Streaming Update of Average Impact Vector
            # NewAvg = OldAvg + (NewVal - OldAvg) / NewCount
            # Streaming Update of Average Impact Vector
            # NewAvg = OldAvg + (NewVal - OldAvg) / NewCount
            old_impact = np.array(edge_data["impact_vector"])
            new_val = np.array(impact_vector)

            # --- FIX: Safe Dimension Alignment ---
            if old_impact.shape != new_val.shape:
                max_len = max(old_impact.shape[0], new_val.shape[0])
                if old_impact.shape[0] < max_len:
                    log(f"[MOSAN] Auto-expanding stored impact vector from {old_impact.shape[0]} to {max_len}")
                    old_impact = np.pad(old_impact, (0, max_len - old_impact.shape[0]))
                if new_val.shape[0] < max_len:
                    # New vector is smaller (legacy tool?), pad it
                    new_val = np.pad(new_val, (0, max_len - new_val.shape[0]))
            # -------------------------------------

            updated_impact = old_impact + (new_val - old_impact) / edge_data["visits"]
            
            edge_data["impact_vector"] = updated_impact.tolist()
            
            # Update latest arguments (So the user always sees the most recent valid configuration)
            if full_arguments:
                 edge_data["full_arguments"] = full_arguments

            # Keep log of raw rewards for deeper analysis (optional, capped size)
            if len(edge_data["raw_rewards"]) < 50:
                edge_data["raw_rewards"].append(impact_vector)
                
        else:
            self.graph.add_edge(
                anchor_from,
                anchor_to,
                key=action_name,
                action=action_name,
                visits=1,
                impact_vector=impact_vector,
                raw_rewards=[impact_vector],
                full_arguments=full_arguments or {},
                q_vector=impact_vector # Init Q with R
            )
            log(f"[MOSAN] Created Link: {anchor_from} -> {anchor_to} via {action_name}")

        # 3. Async Q-Learning Update (Backpropagation of Value)
        self._update_q_values(anchor_from, anchor_to, action_name, impact_vector)

        # 4. Log recent edge update for in-session saturation detectors (Hawk slope, Sage variance).
        try:
            final_q = self.graph.edges[anchor_from, anchor_to, action_name].get("q_vector", impact_vector)
            self._recent_edges_log.append({
                "q_vector": list(final_q),
                "timestamp": time.time(),
                "source": anchor_from,
                "target": anchor_to,
                "action": action_name,
            })
        except Exception:
            pass  # Logging is best-effort; never break the main update path.

        self.save()

    def _update_q_values(self, node_from: str, node_to: str, action_key: str, reward_vector: List[float]):
        """
        Performs an asynchronous Bellman update on the edge (node_from -> node_to).
        Q(s,a) = (1-alpha) * Q(s,a) + alpha * (R + gamma * Max(Q(s', a')))
        """
        ALPHA = 0.1
        GAMMA = 0.9
        
        try:
            # 1. Get Current Q
            edge_data = self.graph.edges[node_from, node_to, action_key]
            current_q = np.array(edge_data.get("q_vector", reward_vector)) # Default to R if missing
            
            # 2. Calculate V(s') = Max_a' Q(s', a') [Element-wise]
            # We look at all outgoing edges from node_to
            downstream_vectors = []
            if self.graph.has_node(node_to):
                for _, _, data in self.graph.edges(node_to, data=True):
                    # Use Q if available, else Impact
                    q = data.get("q_vector", data.get("impact_vector", [0]*5))
                    downstream_vectors.append(q)
            
            if downstream_vectors:
                # Element-wise Max across all downstream actions (Optimistic Bound)
                # Ensure all are same length, pad if needed
                max_len = max(len(v) for v in downstream_vectors)
                clean_vectors = [
                    np.pad(v, (0, max_len - len(v))) if len(v) < max_len else np.array(v)
                    for v in downstream_vectors
                ]
                v_s_prime = np.max(clean_vectors, axis=0)
                
                # Align dimensions with current_q
                if len(v_s_prime) > len(current_q):
                    v_s_prime = v_s_prime[:len(current_q)]
                elif len(v_s_prime) < len(current_q):
                     v_s_prime = np.pad(v_s_prime, (0, len(current_q) - len(v_s_prime)))
            else:
                # Terminal state or no outgoing edges
                v_s_prime = np.zeros_like(current_q)
            
            # 3. Bellman Equation
            reward = np.array(reward_vector)
            # Adaptive Memory Expansion (Scientific Validity Fix)
            # If the new Reward vector has more dimensions (new objective added), 
            # we expand the memory (current_q) instead of discarding data.
            if len(reward) > len(current_q):
                 log(f"[MOSAN] Adaptive Expansion: Upgrading edge memory from {len(current_q)}D to {len(reward)}D")
                 padding = len(reward) - len(current_q)
                 current_q = np.pad(current_q, (0, padding), 'constant')
                 
            # If the Reward vector is smaller (e.g., using a subset of metrics), 
            # we pad the reward to match memory.
            elif len(reward) < len(current_q):
                 reward = np.pad(reward, (0, len(current_q) - len(reward)), 'constant')
                 
            # Recalculate v_s_prime with potential new dimensions
            # (We need to ensure v_s_prime matches the potentially expanded current_q)
            if len(v_s_prime) < len(current_q):
                 v_s_prime = np.pad(v_s_prime, (0, len(current_q) - len(v_s_prime)), 'constant')
            elif len(v_s_prime) > len(current_q):
                 v_s_prime = v_s_prime[:len(current_q)] # Should not happen if logic is consistent
                 
            new_q = (1 - ALPHA) * current_q + ALPHA * (reward + GAMMA * v_s_prime)
            
            # 4. Store
            edge_data["q_vector"] = new_q.tolist()
            # log(f"[MOSAN] Q-Update: {node_from}->{node_to}. OldQ0={current_q[0]:.2f}, NewQ0={new_q[0]:.2f}")
            
        except Exception as e:
            log(f"[MOSAN] Q-Learning Warning: {e}")


    def get_available_transitions(self, state_vector: List[float], architecture: str) -> List[Dict]:
        """
        Returns all known transitions (actions) from the anchor nearest to the current state.
        
        Returns:
            List[Dict]: List of edge data dictionaries containing:
                - action (tool name)
                - target_node (ID)
                - impact_vector
                - visits
                - raw_rewards
                - full_arguments
        """
        if not self.graph:
            return []
            
        # 1. Find Anchor (Simplified KNN, k=1)
        vec = np.array(state_vector).flatten()
        best_node = None
        min_dist = float('inf')
        
        # Optimization: Scan only nodes with same architecture
        for node, data in self.graph.nodes(data=True):
             if data.get("architecture") != architecture:
                 continue
                 
             stored_vec = np.array(data["vector"])
             
             # STATE SPACE EVOLUTION FIX: 
             # If dimensions differ, pad for comparison
             if vec.shape != stored_vec.shape:
                 max_len = max(len(vec), len(stored_vec))
                 v1 = np.pad(vec, (0, max_len - len(vec)))
                 v2 = np.pad(stored_vec, (0, max_len - len(stored_vec)))
                 dist = np.linalg.norm(v1 - v2)
             else:
                 dist = np.linalg.norm(vec - stored_vec)
             
             if dist < min_dist:
                 min_dist = dist
                 best_node = node
        
        # Relaxed threshold for querying (we can offer advice even if slightly far)
        if not best_node or min_dist > SIMILARITY_THRESHOLD * 2.0:
            return []
            
        # 2. Get Outgoing Edges
        transitions = []
        if self.graph.has_node(best_node):
            for u, v, key, data in self.graph.edges(best_node, data=True, keys=True):
                transitions.append({
                    "action": key,
                    "target_node": v,
                    "impact_vector": data.get("impact_vector", []),
                    "visits": data.get("visits", 0),
                    "raw_rewards": data.get("raw_rewards", []),
                    "full_arguments": data.get("full_arguments", {}),
                    "q_vector": data.get("q_vector", None)
                })
                
        return transitions

    def get_node_data(self, node_id: str) -> Dict:
        """Returns data for a specific node (vector, visits)."""
        if self.graph.has_node(node_id):
            return self.graph.nodes[node_id]
        return {}

    def get_strategic_graph(self) -> dict:
        """Returns JSON-serializable graph."""
        if not self.graph:
            return {"nodes": [], "edges": []}
        return nx.node_link_data(self.graph)

    # ============================================================================
    # PATHFINDING METHODS (Hybrid Graph Intelligence System)
    # ============================================================================
    
    def get_high_value_states(self, top_k: int = 5, min_q_value: float = 0.5) -> List[str]:
        """
        Finds anchor nodes with highest outgoing Q-values.
        These represent "valuable" states that the agent should aim for.
        
        Args:
            top_k: Number of high-value states to return
            min_q_value: Minimum Q-value threshold
            
        Returns:
            List of node IDs sorted by average outgoing Q-value
        """
        if not self.graph or not self.graph.nodes:
            return []
        
        node_scores = []
        
        for node in self.graph.nodes():
            # Get all outgoing edges and their Q-values
            q_values = []
            for _, _, data in self.graph.edges(node, data=True):
                q_vec = data.get("q_vector", data.get("impact_vector", []))
                if q_vec and len(q_vec) > 0:
                    # Use first element (accuracy improvement) as primary metric
                    q_values.append(q_vec[0])  
            
            if q_values:
                avg_q = np.mean(q_values)
                if avg_q >= min_q_value:
                    node_scores.append((node, avg_q))
        
        # Sort by average Q-value descending
        node_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [node_id for node_id, _ in node_scores[:top_k]]

    def get_node_arch(self, node_id: str) -> str:
        """Returns the architecture name for a node, stripping the UUID suffix if needed."""
        return self.graph.nodes.get(node_id, {}).get(
            "architecture", str(node_id).rsplit("_", 1)[0]
        )

    def get_node_avg_q(self, node_id: str) -> Optional[float]:
        """Returns the average outgoing Q-value for a node, or None if no edges exist."""
        q_vals = []
        for _, _, data in self.graph.edges(node_id, data=True):
            q_vec = data.get("q_vector", data.get("impact_vector", []))
            if q_vec:
                q_vals.append(q_vec[0])
        return float(np.mean(q_vals)) if q_vals else None

    # =====================================================================
    # Saturation detector helpers (per-lens signal metrics).
    # All methods exclude ephemeral nodes/edges to keep ghost-injected
    # cross-dataset wisdom from skewing in-session saturation signals.
    # =====================================================================

    def _non_ephemeral_nodes(self) -> List[str]:
        return [n for n, d in self.graph.nodes(data=True) if not d.get("ephemeral")]

    def _non_ephemeral_edges(self) -> List[Tuple[str, str, Dict]]:
        return [
            (u, v, d)
            for u, v, d in self.graph.edges(data=True)
            if not d.get("ephemeral")
            and not self.graph.nodes[u].get("ephemeral")
            and not self.graph.nodes[v].get("ephemeral")
        ]

    def get_graph_novelty_percentile(self, visits: int) -> float:
        """
        Returns the fraction of non-ephemeral nodes with visit count <= `visits`.
        High percentile means the reference node is among the *most-visited* (least novel).
        Returns 0.0 on a graph with fewer than MIN_NODES_FOR_PERCENTILE nodes (cold-graph guard).

        Used by: Explorer lens saturation detector.
        """
        nodes = self._non_ephemeral_nodes()
        if len(nodes) < MIN_NODES_FOR_PERCENTILE:
            return 0.0
        visit_counts = [self.graph.nodes[n].get("visits", 0) for n in nodes]
        if not visit_counts:
            return 0.0
        leq = sum(1 for vc in visit_counts if vc <= visits)
        return float(leq) / float(len(visit_counts))

    def get_recent_edges(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Returns the last `n` edges that received an update in the current session.
        Each entry: {"q_vector": [...], "timestamp": ..., "source": ..., "target": ..., "action": ...}.
        Empty list on a fresh session (in-memory only, does not persist across restarts).

        Used by: Hawk slope, Sage variance.
        """
        if n <= 0:
            return []
        # deque is appended chronologically; take last n preserving order
        items = list(self._recent_edges_log)
        return items[-n:]

    def get_efficiency_coverage(self) -> float:
        """
        Returns fraction of non-ephemeral edges with at least one non-zero efficiency
        component (q_vector[Q_VEC_LATENCY] != 0 OR q_vector[Q_VEC_MEMORY] != 0).
        Returns 1.0 if there are zero edges (vacuously fully covered — nothing to profile).

        Used by: Engineer lens saturation detector.
        """
        edges = self._non_ephemeral_edges()
        if not edges:
            return 1.0
        covered = 0
        for _, _, d in edges:
            q_vec = d.get("q_vector") or d.get("impact_vector") or []
            try:
                lat = q_vec[Q_VEC_LATENCY] if len(q_vec) > Q_VEC_LATENCY else 0
                mem = q_vec[Q_VEC_MEMORY] if len(q_vec) > Q_VEC_MEMORY else 0
            except (TypeError, IndexError):
                lat, mem = 0, 0
            if (lat or 0) != 0 or (mem or 0) != 0:
                covered += 1
        return float(covered) / float(len(edges))

    def get_lens_signal_metric(self, lens_name: str = "", recent_window: int = 10) -> Dict[str, float]:
        """
        Single dispatcher that returns all four saturation metrics, used by ParetoRanker.

        The `lens_name` argument is accepted for API symmetry but currently all four
        metrics are computed unconditionally — the cost is negligible and the caller
        decides which one to read.

        Returns:
            {
              "novelty_percentile": float,    # for Explorer (overall graph penetration)
              "accuracy_slope":     float,    # for Hawk (slope of recent q[acc])
              "stability_std":      float,    # for Sage (std of recent q[overfit])
              "efficiency_coverage":float,    # for Engineer
              "n_nodes":            int,
              "n_recent_edges":     int,
            }
        """
        nodes = self._non_ephemeral_nodes()
        n_nodes = len(nodes)

        # Aggregate "novelty pressure" of the graph: percentile of the median visit count
        # against itself = 1.0 if uniformly visited; here we report fraction of nodes
        # that are no longer novel (visits >= median). Caller passes a specific
        # target's visit count to get a target-specific percentile via get_graph_novelty_percentile.
        if n_nodes >= MIN_NODES_FOR_PERCENTILE:
            visit_counts = [self.graph.nodes[n].get("visits", 0) for n in nodes]
            median_v = float(np.median(visit_counts)) if visit_counts else 0.0
            novelty_percentile = self.get_graph_novelty_percentile(int(median_v))
        else:
            novelty_percentile = 0.0

        # Recent-edge metrics (Hawk slope, Sage variance)
        recent = self.get_recent_edges(recent_window)
        n_recent = len(recent)

        accuracy_slope = 0.0
        stability_std = 0.0
        if n_recent >= 2:
            acc_vals = []
            overfit_vals = []
            for entry in recent:
                qv = entry.get("q_vector") or []
                if len(qv) > Q_VEC_ACC:
                    acc_vals.append(float(qv[Q_VEC_ACC]))
                if len(qv) > Q_VEC_OVERFIT:
                    overfit_vals.append(float(qv[Q_VEC_OVERFIT]))
            if len(acc_vals) >= 2:
                try:
                    accuracy_slope = float(np.polyfit(np.arange(len(acc_vals)), acc_vals, 1)[0])
                except Exception:
                    accuracy_slope = 0.0
            if len(overfit_vals) >= 2:
                stability_std = float(np.std(overfit_vals))

        efficiency_coverage = self.get_efficiency_coverage()

        return {
            "novelty_percentile": float(novelty_percentile),
            "accuracy_slope": float(accuracy_slope),
            "stability_std": float(stability_std),
            "efficiency_coverage": float(efficiency_coverage),
            "n_nodes": int(n_nodes),
            "n_recent_edges": int(n_recent),
        }

    def get_local_neighborhood(self, state_vector: List[float], architecture: str, k_hops: int = 2) -> Dict:
        """
        Returns the k-hop neighborhood subgraph from the nearest anchor node.
        Used for local graph analysis by LLM.
        
        Args:
            state_vector: Current state vector
            architecture: Architecture type to match
            k_hops: Number of hops to traverse (default 2)
            
        Returns:
            Dict with 'nodes' and 'edges' representing the local subgraph
        """
        if not self.graph or not self.graph.nodes:
            return {"nodes": [], "edges": []}
        
        # Find nearest anchor (reuse existing logic)
        vec = np.array(state_vector).flatten()
        best_node = None
        min_dist = float('inf')
        
        for node, data in self.graph.nodes(data=True):
            if data.get("architecture") != architecture:
                continue
            
            stored_vec = np.array(data["vector"])
            
            if vec.shape != stored_vec.shape:
                max_len = max(len(vec), len(stored_vec))
                v1 = np.pad(vec, (0, max_len - len(vec)))
                v2 = np.pad(stored_vec, (0, max_len - len(stored_vec)))
                dist = np.linalg.norm(v1 - v2)
            else:
                dist = np.linalg.norm(vec - stored_vec)
            
            if dist < min_dist:
                min_dist = dist
                best_node = node
        
        if not best_node:
            return {"nodes": [], "edges": []}
        
        # Get k-hop neighborhood using BFS        
        neighbors = {best_node}
        current_layer = {best_node}
        
        for _ in range(k_hops):
            next_layer = set()
            for node in current_layer:
                # Add successors (outgoing edges)
                next_layer.update(self.graph.successors(node))
                # Optionally: add predecessors (incoming edges) for context
                # next_layer.update(self.graph.predecessors(node))
            neighbors.update(next_layer)
            current_layer = next_layer
            
            if not current_layer:
                break
        
        # Extract subgraph
        subgraph = self.graph.subgraph(neighbors)
        
        # Convert to simple dict format
        nodes_data = []
        for node in subgraph.nodes():
            node_data = subgraph.nodes[node].copy()
            node_data["id"] = node
            nodes_data.append(node_data)
        
        edges_data = []
        for u, v, key, data in subgraph.edges(data=True, keys=True):
            edge_dict = data.copy()
            edge_dict["from"] = u
            edge_dict["to"] = v
            edge_dict["action"] = key
            edges_data.append(edge_dict)
        
        return {"nodes": nodes_data, "edges": edges_data}
    
    def find_path(self, state_vector: List[float], architecture: str, goal_states: List[str] = None, max_length: int = 10) -> List[Dict]:
        """
        Finds the optimal path from current state to a high-value goal state using A* algorithm.
        Uses Q-values as edge weights (higher Q = lower cost = preferred path).
        
        Args:
            state_vector: Current state vector
            architecture: Architecture type to match
            goal_states: Optional list of goal node IDs. If None, uses get_high_value_states()
            max_length: Maximum path length
            
        Returns:
            List of dicts, each containing:
                - from_node: Starting node ID
                - to_node: Ending node ID  
                - action: Action name
                - q_value: Expected Q-value
                - impact_vector: Expected impact
                - full_arguments: Tool arguments for replay
        """
        if not self.graph or not self.graph.nodes:
            return []
        
        # 1. Find starting node (nearest anchor)
        vec = np.array(state_vector).flatten()
        start_node = None
        min_dist = float('inf')
        
        for node, data in self.graph.nodes(data=True):
            if data.get("architecture") != architecture:
                continue
            
            stored_vec = np.array(data["vector"])
            
            if vec.shape != stored_vec.shape:
                max_len_dim = max(len(vec), len(stored_vec))
                v1 = np.pad(vec, (0, max_len_dim - len(vec)))
                v2 = np.pad(stored_vec, (0, max_len_dim - len(stored_vec)))
                dist = np.linalg.norm(v1 - v2)
            else:
                dist = np.linalg.norm(vec - stored_vec)
            
            if dist < min_dist:
                min_dist = dist
                start_node = node
        
        if not start_node:
            return []
        
        # 2. Determine goal states
        if not goal_states:
            goal_states = self.get_high_value_states(top_k=3)
        
        if not goal_states:
            return []
        
        # 3. A* pathfinding
        # We want to MAXIMIZE Q-value, so cost = -Q (lower cost = higher Q)
        def heuristic(node, target):
            """Estimate remaining cost to goal (negative average Q)"""
            q_vals = []
            # Correctly iterate outgoing edges
            for _, _, data in self.graph.out_edges(node, data=True):
                q_vec = data.get("q_vector", data.get("impact_vector", []))
                if q_vec and len(q_vec) > 0:
                    q_vals.append(q_vec[0])
            
            if q_vals:
                return -np.mean(q_vals)  # Higher Q = lower cost
            return 0
        
        best_path = None
        best_path_value = float('-inf')
        
        # Try to find path to each goal state
        for goal in goal_states:
            try:
                # Use NetworkX A* with custom weight function
                def weight(u, v, d):
                    """Edge weight = -Q_value (to maximize Q via minimizing cost)"""
                    q_vec = d.get("q_vector", d.get("impact_vector", [0]))
                    if q_vec and len(q_vec) > 0:
                        return -q_vec[0]
                    return 0
                
                path_nodes = nx.astar_path(
                    self.graph, 
                    start_node, 
                    goal,
                    heuristic=heuristic,
                    weight=weight
                )
                
                if len(path_nodes) > max_length:
                    continue
                
                # Calculate total path Q-value
                path_value = 0
                for i in range(len(path_nodes) - 1):
                    u, v = path_nodes[i], path_nodes[i+1]
                    # Get edge with highest Q between u and v
                    best_q = float('-inf')
                    
                    # Handle MultiDiGraph access properly
                    if self.graph.has_edge(u, v):
                        edges = self.graph[u][v]
                        # edges is a dict of key->attr
                        for key, data in edges.items():
                            q_vec = data.get("q_vector", data.get("impact_vector", []))
                            if q_vec and len(q_vec) > 0 and q_vec[0] > best_q:
                                best_q = q_vec[0]
                    
                    if best_q != float('-inf'):
                        path_value += best_q
                
                if path_value > best_path_value:
                    best_path_value = path_value
                    best_path = path_nodes
                    
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        
        if not best_path or len(best_path) < 2:
            return []
        
        # 4. Convert path to action list
        path_actions = []
        for i in range(len(best_path) - 1):
            from_node = best_path[i]
            to_node = best_path[i + 1]
            
            # Find best edge between these nodes (highest Q-value)
            best_edge = None
            best_q = float('-inf')
            
            if self.graph.has_edge(from_node, to_node):
                edges = self.graph[from_node][to_node]
                for key, data in edges.items():
                    q_vec = data.get("q_vector", data.get("impact_vector", []))
                    if q_vec and len(q_vec) > 0 and q_vec[0] > best_q:
                        best_q = q_vec[0]
                        best_edge = (key, data)
            
            if best_edge:
                action_name, edge_data = best_edge
                path_actions.append({
                    "from_node": from_node,
                    "to_node": to_node,
                    "action": action_name,
                    "q_value": edge_data.get("q_vector", [0])[0] if edge_data.get("q_vector") else 0,
                    "impact_vector": edge_data.get("impact_vector", []),
                    "visits": edge_data.get("visits", 0),
                    "full_arguments": edge_data.get("full_arguments", {})
                })
        
        return path_actions
