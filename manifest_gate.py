"""
Manifest Gate — prevents redundant manifest creation by scanning
datasets_metadata.json for siblings that already have the requested
representation physically generated on disk with equivalent transformation
parameters.

Called from inside `create_dataset_manifest` BEFORE manifest creation, so it
covers both the LLM-driven Decider flow and the Python-driven Innovation Team
flow with a single intercept.

Behaviour:
- raw_data_path unresolvable → HALT (set halt_request flag, return blocked).
- Compatible sibling found → return blocked_use_existing with effective_manifest_name.
- No sibling → return None (caller proceeds with normal creation).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from ui_logger import log


# ---------------------------------------------------------------------------
# Tensor-defining params per representation_type.
#
# Two manifests with the same rep_type but different values for these params
# produce DIFFERENT tensors (either different shape or different content).
# Hence sibling redirect is only safe when ALL listed params match.
#
# Defaults below mirror the data_pipeline_worker fallbacks (data_pipeline_worker.py:533-543)
# — these are the values that actually shape the .npy files on disk when the
# generator is invoked without explicit params.
# ---------------------------------------------------------------------------
TENSOR_DEFINING_PARAMS: Dict[str, Tuple[str, ...]] = {
    "3D_DYNAMIC_GAF":   ("video_num_segments", "video_gaf_image_size", "gaf_method"),
    "3D_DYNAMIC_CWT":   ("video_num_segments", "video_gaf_image_size"),
    "3D_WAVELET_CWT":   ("video_num_segments", "video_gaf_image_size"),
    "3D_GAF_VIDEO":     ("video_num_segments", "video_gaf_image_size"),
    "3D_VIDEO":         ("video_num_segments",),
    "2D_GAF":           ("gaf_image_size", "gaf_method"),
    "2D_CWT_SCALOGRAM": ("gaf_image_size",),
    "2D_SPECTROGRAM":   ("n_fft", "hop_length"),
    "2D_GENERIC_IMAGE": (),  # no transformation
    "2D_IMAGE":         (),
    "1D_CNN":           (),  # no transformation
}

# Worker-level defaults used when params are missing in the request.
# Source of truth: processed_data/scripts_worker/data_pipeline_worker.py:533-543
WORKER_DEFAULTS: Dict[str, Any] = {
    "video_num_segments": 60,
    "video_gaf_image_size": 64,
    "gaf_method": "summation",
    "gaf_image_size": 64,
    "n_fft": 512,
    "hop_length": 64,
}


# ---------------------------------------------------------------------------
# Public entry point — called from create_dataset_manifest in tools.py
# ---------------------------------------------------------------------------
def evaluate_manifest_request(
    requested_name: str,
    requested_params: Dict[str, Any],
    metadata: Dict[str, Any],
    thread_safe_state: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Decide whether to allow, block-and-redirect, or halt a manifest creation.

    Returns:
        None — caller may proceed with normal creation (no compatible sibling).
        dict — caller MUST return this dict immediately to its caller.
               Possible status values:
                 - "blocked_user_input_required" (raw_data_path unresolvable; halt_request set)
                 - "blocked_use_existing" (sibling found; effective_manifest_name provided)
    """
    rep_type = (requested_params.get("representation_type") or "").strip()
    if not rep_type:
        # No rep_type specified — let create_dataset_manifest's own validation
        # produce its existing error. Don't double up.
        return None

    raw_data_path = _resolve_raw_data_path(thread_safe_state, requested_params)
    if not raw_data_path:
        # HARD HALT — refuse to guess a data source.
        return _halt_for_user_input(
            thread_safe_state=thread_safe_state,
            requested_name=requested_name,
            rep_type=rep_type,
        )

    sibling = _find_sibling_manifest(
        metadata=metadata,
        rep_type=rep_type,
        raw_data_path=raw_data_path,
        requested_params=requested_params,
    )

    if sibling is None:
        return None  # caller proceeds with normal creation

    return _block_with_redirect(
        requested_name=requested_name,
        sibling_name=sibling,
        sibling_meta=metadata[sibling],
        rep_type=rep_type,
    )


# ---------------------------------------------------------------------------
# Sibling lookup — match on rep_type + raw_data_path + tensor-defining params
# ---------------------------------------------------------------------------
def _find_sibling_manifest(
    metadata: Dict[str, Any],
    rep_type: str,
    raw_data_path: str,
    requested_params: Dict[str, Any],
) -> Optional[str]:
    """
    Scan metadata for a manifest that:
      1. Has rep_type registered in generated_representations
      2. Was built from the same raw_data_path (normalized basename match)
      3. Has identical tensor-defining params (with WORKER_DEFAULTS as fallback
         on both sides for missing keys)
      4. Has the rep file physically present on disk (guard against stale metadata)

    Returns the best sibling's name, or None.
    """
    rep_norm = rep_type.upper().strip()
    src_norm = _normalize_data_source(raw_data_path)
    relevant_keys = TENSOR_DEFINING_PARAMS.get(rep_norm, ())

    candidates: list[str] = []

    for name, meta in metadata.items():
        if not isinstance(meta, dict):
            continue

        gen_reps = meta.get("generated_representations") or {}
        # Case-insensitive match on rep keys
        gen_reps_upper = {k.upper().strip(): v for k, v in gen_reps.items()}
        if rep_norm not in gen_reps_upper:
            continue

        meta_params = meta.get("params") or {}
        meta_src = meta_params.get("raw_data_path", "")
        if _normalize_data_source(meta_src) != src_norm:
            continue

        # Tensor-defining params match (with worker defaults as fallback)
        if not _params_equivalent(requested_params, meta_params, relevant_keys):
            continue

        # Physical file existence guard
        rep_path = gen_reps_upper.get(rep_norm)
        if not rep_path:
            continue
        # The .npy may be at the registered path or in a sibling subfolder.
        if not _rep_file_exists(rep_path, meta, rep_norm):
            log(
                f"[ManifestGate] Skipping '{name}': rep '{rep_norm}' registered "
                f"but file missing on disk ({rep_path})."
            )
            continue

        candidates.append(name)

    if not candidates:
        return None

    # Tie-break: prefer the manifest with the richest cache (most reps generated).
    candidates.sort(
        key=lambda n: len(metadata[n].get("generated_representations") or {}),
        reverse=True,
    )
    return candidates[0]


def _params_equivalent(
    requested: Dict[str, Any],
    sibling: Dict[str, Any],
    relevant_keys: Tuple[str, ...],
) -> bool:
    """All tensor-defining params must match. Missing keys fall back to WORKER_DEFAULTS."""
    for k in relevant_keys:
        req_val = requested.get(k, WORKER_DEFAULTS.get(k))
        sib_val = sibling.get(k, WORKER_DEFAULTS.get(k))
        if req_val != sib_val:
            return False
    return True


def _normalize_data_source(path: str) -> str:
    """
    Compare data sources by basename + lowercase. We avoid full-path comparison
    because PERSISTENT_PATHS may differ between cluster nodes / Docker / host.
    Two manifests pointing at .../raw_data/NPY_nature_30_classes are equivalent
    regardless of the absolute prefix.
    """
    if not path:
        return ""
    return os.path.basename(os.path.normpath(path)).lower()


def _rep_file_exists(rep_path: str, meta: Dict[str, Any], rep_norm: str) -> bool:
    """Verify the registered rep file is physically present (or its sibling)."""
    if rep_path and os.path.exists(rep_path):
        return True
    # Fallback: check the manifest's path/<REP>_data.npy
    base_path = meta.get("path", "")
    if base_path:
        candidate = os.path.join(base_path, f"{rep_norm}_data.npy")
        if os.path.exists(candidate):
            return True
    return False


# ---------------------------------------------------------------------------
# Halt mechanism — for raw_data_path unresolvable
# ---------------------------------------------------------------------------
def _resolve_raw_data_path(
    thread_safe_state: Dict[str, Any],
    requested_params: Dict[str, Any],
) -> Optional[str]:
    """Resolve from explicit param first, then from session state."""
    explicit = (requested_params.get("raw_data_path") or "").strip()
    if explicit:
        return explicit

    src_name = thread_safe_state.get("selected_raw_data_source")
    if not src_name:
        return None

    # Resolve basename to full path under raw_data_dir
    try:
        from state_manager import PERSISTENT_PATHS
        raw_dir = PERSISTENT_PATHS.get("raw_data_dir", "processed_data/raw_data")
    except Exception:
        raw_dir = "processed_data/raw_data"
    return os.path.join(raw_dir, src_name)


def _halt_for_user_input(
    thread_safe_state: Dict[str, Any],
    requested_name: str,
    rep_type: str,
) -> Dict[str, Any]:
    """Set halt_request flag and return a blocking response."""
    halt_payload = {
        "reason": "raw_data_path_unresolvable",
        "requested_manifest": requested_name,
        "requested_rep_type": rep_type,
        "message": (
            f"Cannot create manifest '{requested_name}': raw_data_path is not set "
            f"in session state and was not provided in params. The system requires "
            f"explicit user input to choose the data source. "
            f"Set 'selected_raw_data_source' or pass 'raw_data_path' in params."
        ),
    }
    thread_safe_state["halt_request"] = halt_payload
    log(f"[ManifestGate] HALT requested: {halt_payload['message']}")

    return {
        "status": "blocked_user_input_required",
        "halt": True,
        "message": (
            f"[HALT] {halt_payload['message']} "
            f"DO NOT retry create_dataset_manifest with another name — "
            f"the gate will keep blocking until raw_data_path is resolved by the user."
        ),
    }


# ---------------------------------------------------------------------------
# Block + redirect
# ---------------------------------------------------------------------------
def _block_with_redirect(
    requested_name: str,
    sibling_name: str,
    sibling_meta: Dict[str, Any],
    rep_type: str,
) -> Dict[str, Any]:
    """Return a blocking response that forces the caller to use the sibling."""
    relevant_keys = TENSOR_DEFINING_PARAMS.get(rep_type.upper().strip(), ())
    sibling_params = sibling_meta.get("params") or {}
    transform_view = {k: sibling_params.get(k, WORKER_DEFAULTS.get(k)) for k in relevant_keys}

    msg = (
        f"[BLOCKED] Refused to create new manifest '{requested_name}'. "
        f"An equivalent manifest '{sibling_name}' already exists with IDENTICAL "
        f"transformation parameters and the data is physically generated on disk. "
        f"\n\nMANDATORY ACTION: In your next tool call, use manifest_name='{sibling_name}'. "
        f"Equivalent params: {transform_view}. "
        f"\n\nDO NOT call create_dataset_manifest again with a different name "
        f"unless you change the representation_type OR the tensor-defining params "
        f"({list(relevant_keys) if relevant_keys else 'none for this rep_type'}). "
        f"The gate will block any equivalent request."
    )
    log(
        f"[ManifestGate] Hard redirect: '{requested_name}' -> '{sibling_name}' "
        f"(rep={rep_type}, params={transform_view}). Saved data regeneration."
    )

    return {
        "status": "blocked_use_existing",
        "blocking": True,
        "redirected": True,
        "effective_manifest_name": sibling_name,
        "sibling_metadata": {
            "name": sibling_name,
            "raw_data_path": sibling_params.get("raw_data_path"),
            "representation_type": rep_type,
            "transform_params": transform_view,
        },
        "message": msg,
    }
