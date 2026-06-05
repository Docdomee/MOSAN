"""
finetune_training_worker.py
============================
Worker for the two-phase Ho-et-al protocol:

  Phase 1 — Pretraining (multi-seed)
    Trains the chosen architecture on {rep}_data.npy for each seed,
    validating on {rep}_X_test_heldout.npy (true HO test set).
    Keeps the seed with the highest val_accuracy.

  Phase 2 — Adaptive fine-tuning (multi-seed ensemble)
    Loads {rep}_X_finetune.npy + labels_finetune.npy (pre-generated
    by data_pipeline_worker._try_generate_finetune).
    Runs 3 finetune seeds with randomised splits, averages softmax outputs
    on the shifted test set (soft voting). Freezing depth controlled by
    ft_unfreeze_last_n (0 = head only, N = unfreeze last N backbone layers).

  Phase 3 — Evaluation
    Per-spectrum accuracy on {rep}_X_test_heldout.npy
    Per-class breakdown
    Always saves the final model to disk.

This worker is invoked exclusively by tools.run_training_trial_with_finetune
via the Celery task cluster.tasks.run_finetune_training_worker.
It is NOT meant to be used for hyperparameter exploration — that role
belongs to training_worker.py (which runs CV).
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback

import numpy as np

# Make worker modules importable
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPTS_DIR, "..", ".."))
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "processed_data", "custom_architectures"))

from state_manager import PERSISTENT_PATHS


# ---------------------------------------------------------------------------
# Model loader (mirrors training_worker.py)
# ---------------------------------------------------------------------------

def load_model_builder(custom_architecture_file, experiment_name, representation_type):
    """Return the build_model callable, either from custom file or MODEL_BUILDERS."""
    if custom_architecture_file:
        filepath = os.path.join(PERSISTENT_PATHS["custom_architectures_dir"], custom_architecture_file)
        if not os.path.exists(filepath) and not filepath.endswith(".py"):
            filepath += ".py"
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Custom architecture file not found: {filepath}")
        # Ghost-execution guard. Three layers, all needed:
        #  (1) sys.modules.pop  — evict any in-process cached module object
        #  (2) invalidate_caches — clear sys.path_importer_cache (path finders)
        #  (3) delete __pycache__/<mod>.* — Python's SourceFileLoader checks the
        #      .pyc mtime against source mtime; when write_architecture_file rewrites
        #      the source within the same FS-clock second (common on cluster NFS),
        #      mtimes match and the stale .pyc is served. Removing the .pyc forces
        #      re-compile from source on the very next import.
        import glob as _glob
        mod_name = custom_architecture_file.replace(".py", "")
        sys.modules.pop(mod_name, None)
        importlib.invalidate_caches()
        _pycache_dir = os.path.join(os.path.dirname(filepath), "__pycache__")
        if os.path.isdir(_pycache_dir):
            for _pyc in _glob.glob(os.path.join(_pycache_dir, mod_name + ".*")):
                try:
                    os.remove(_pyc)
                except OSError:
                    pass
        spec = importlib.util.spec_from_file_location(mod_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.build_model

    from model_factory import MODEL_BUILDERS
    builder = MODEL_BUILDERS.get(experiment_name) or MODEL_BUILDERS.get(representation_type)
    if builder is None:
        raise ValueError(
            f"No model builder for experiment='{experiment_name}' or representation='{representation_type}'")
    return builder


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_pretrain_data(data_path_root, representation_type):
    """Returns X_train, X_test (HO), y_train, y_test, label_encoder, num_classes."""
    from sklearn.preprocessing import LabelEncoder

    X_train = np.load(os.path.join(data_path_root, f"{representation_type}_data.npy")).astype("float32")
    y_train = np.load(os.path.join(data_path_root, "labels.npy"), allow_pickle=True)

    test_x_path = os.path.join(data_path_root, f"{representation_type}_X_test_heldout.npy")
    test_y_path = os.path.join(data_path_root, "labels_test_heldout.npy")
    if not (os.path.exists(test_x_path) and os.path.exists(test_y_path)):
        raise FileNotFoundError(
            f"HO test set missing: {test_x_path}. "
            f"This tool requires an explicit held-out test set.")
    X_test = np.load(test_x_path).astype("float32")
    y_test = np.load(test_y_path, allow_pickle=True)

    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_test]))
    y_train_int = le.transform(y_train)
    y_test_int = le.transform(y_test)
    num_classes = len(le.classes_)

    print(f"[Data] Train: {X_train.shape}  Test(HO): {X_test.shape}  classes: {num_classes}")
    return X_train, X_test, y_train_int, y_test_int, le, num_classes


def load_finetune_data(data_path_root, representation_type, le):
    """Returns X_ft, y_ft_int  or  (None, None) if finetune split unavailable."""
    x_path = os.path.join(data_path_root, f"{representation_type}_X_finetune.npy")
    y_path = os.path.join(data_path_root, "labels_finetune.npy")
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        return None, None
    X_ft = np.load(x_path).astype("float32")
    y_ft_raw = np.load(y_path, allow_pickle=True)
    y_ft_int = le.transform(y_ft_raw)
    print(f"[Data] Finetune: {X_ft.shape}")
    return X_ft, y_ft_int


# ---------------------------------------------------------------------------
# Phase 1 — Multi-seed pretrain
# ---------------------------------------------------------------------------

def pretrain_multiseed(model_builder, X_train, y_train_int, X_test, y_test_int,
                        num_classes, hp, seeds, out_dir, run_id):
    import tensorflow as tf
    from tensorflow import keras

    y_train_cat = keras.utils.to_categorical(y_train_int, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test_int, num_classes)

    epochs   = int(hp.get("epochs", 100))
    bs       = int(hp.get("batch_size", 32))
    patience = int(hp.get("patience", 20))

    best_val = -1.0
    best_path = None
    seed_results = []

    for seed in seeds:
        tf.random.set_seed(seed)
        np.random.seed(seed)

        model = model_builder(X_train.shape[1:], num_classes, hp)
        # An uncompiled Functional model has NO `optimizer` attribute (accessing it
        # raises AttributeError, not None) — getattr makes the "needs compile?" check
        # robust to both compiled and uncompiled builders.
        if getattr(model, "optimizer", None) is None:
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=float(hp.get("learning_rate", 1e-3))),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )

        print(f"\n[Phase 1 seed={seed}] epochs={epochs} bs={bs} patience={patience}")
        t0 = time.time()
        history = model.fit(
            X_train, y_train_cat,
            epochs=epochs,
            batch_size=bs,
            validation_data=(X_test, y_test_cat),
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_accuracy",
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1,
                )
            ],
            verbose=2,
        )
        elapsed = time.time() - t0
        val_acc = max(history.history["val_accuracy"])
        print(f"[Phase 1 seed={seed}] Done {elapsed/60:.1f} min — val_accuracy={val_acc:.4f}")

        path = os.path.join(out_dir, f"{run_id}_pretrained_seed{seed}.keras")
        model.save(path)
        seed_results.append({"seed": seed, "val_accuracy": float(val_acc), "path": path})
        if val_acc > best_val:
            best_val = float(val_acc)
            best_path = path

        keras.backend.clear_session()

    return best_path, best_val, seed_results


# ---------------------------------------------------------------------------
# Phase 2 — Full-model fine-tuning (single best seed)
# ---------------------------------------------------------------------------

def finetune_full_model(best_pretrain_path, X_ft, y_ft_int, num_classes,
                         ft_lr, ft_epochs, out_dir, run_id, hp, split_seed=None):
    from tensorflow import keras
    from sklearn.utils.class_weight import compute_class_weight

    y_ft_cat = keras.utils.to_categorical(y_ft_int, num_classes)

    # Use the ENTIRE finetune set to adapt — never carve a validation split out of an
    # already-tiny calibration pool. With ~4 samples/class a held-out val makes
    # val_accuracy pure noise; EarlyStopping(restore_best_weights) then keeps a
    # near-pretrained checkpoint and the finetune no-ops. We instead train for fixed
    # epochs and anneal the LR on the TRAINING loss. The shifted set is the final
    # held-out evaluation (Phase 3) and must never influence training (no leakage).
    print(f"\n[Phase 2] Finetune train={len(X_ft)} (full set — no val carve)  "
          f"seed={split_seed}  lr={ft_lr}")

    # split_seed now only diversifies training randomness (init / batch shuffle) so the
    # multi-seed soft-voting ensemble stays diverse — there is no calib/val split to vary.
    if split_seed is not None:
        keras.utils.set_random_seed(int(split_seed))

    model = keras.models.load_model(best_pretrain_path)

    # Granular freezing: ft_unfreeze_last_n controls how many non-Dense backbone
    # layers (counting from the top) are allowed to adapt.
    # 0  → head only (safest on tiny sets)
    # N  → last N backbone layers + head (progressive unfreezing)
    # -1 / ft_freeze_backbone=False legacy → full unfreeze
    ft_unfreeze_last_n = int(hp.get("ft_unfreeze_last_n", 0))
    # Backward compat: old HP dict with ft_freeze_backbone=False → full unfreeze
    if not hp.get("ft_freeze_backbone", True):
        ft_unfreeze_last_n = len(model.layers)

    non_dense = [l for l in model.layers if "dense" not in l.name.lower()]
    for i, layer in enumerate(non_dense):
        layer.trainable = (ft_unfreeze_last_n > 0 and i >= len(non_dense) - ft_unfreeze_last_n)
    ft_reinitialize_head = int(hp.get("ft_reinitialize_head", 1))
    for layer in model.layers:
        if "dense" in layer.name.lower():
            layer.trainable = True
            if ft_reinitialize_head:
                import tensorflow as tf
                # Randomize the dense layer weights to destroy old pre-training bias
                if hasattr(layer, "kernel_initializer") and hasattr(layer, "kernel"):
                    layer.kernel.assign(layer.kernel_initializer(tf.shape(layer.kernel)))
                if hasattr(layer, "bias_initializer") and hasattr(layer, "bias") and layer.bias is not None:
                    layer.bias.assign(layer.bias_initializer(tf.shape(layer.bias)))

    trainable_count = sum(1 for l in model.layers if l.trainable)
    frozen_count = len(model.layers) - trainable_count
    print(f"[Phase 2] ft_unfreeze_last_n={ft_unfreeze_last_n} → "
          f"{trainable_count} trainable, {frozen_count} frozen.")

    # Always use the exact ft_lr globally. 
    # Downscaling backbone_lr caused catastrophic failure to adapt.
    print(f"[Phase 2] Global Finetune LR={ft_lr:.2e} (Head Reinit={ft_reinitialize_head})")
    optimizer = keras.optimizers.Adam(learning_rate=ft_lr)

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # Class weighting prevents majority-class collapse on imbalanced small sets.
    classes = np.arange(num_classes)
    cw_values = compute_class_weight("balanced", classes=classes, y=y_ft_int)
    class_weight = dict(enumerate(cw_values))

    t0 = time.time()
    history = model.fit(
        X_ft, y_ft_cat,
        epochs=ft_epochs,
        batch_size=min(16, len(X_ft)),
        class_weight=class_weight,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(
                monitor="loss",
                factor=0.5,
                patience=15,
                min_lr=1e-7,
                verbose=1,
            )
        ],
        verbose=2,
    )
    elapsed = time.time() - t0
    # No validation set by design — report final TRAINING accuracy. It is used only to
    # pick which single model reports HO accuracy; the Phase 3 shifted ensemble uses all
    # seeds regardless, so this selector never touches the held-out shifted set.
    final_train_acc = float(history.history["accuracy"][-1])
    print(f"[Phase 2] Done {elapsed/60:.1f} min — final train_accuracy={final_train_acc:.4f}")

    ft_path = os.path.join(out_dir, f"{run_id}_seed{split_seed}_finetuned.keras")
    model.save(ft_path)
    return model, final_train_acc, ft_path


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X_test, y_test_int, le, num_classes):
    from sklearn.metrics import f1_score, matthews_corrcoef
    y_pred      = model.predict(X_test, batch_size=min(32, len(X_test)), verbose=0).argmax(axis=1)
    acc         = float(np.mean(y_pred == y_test_int))
    f1_macro    = float(f1_score(y_test_int, y_pred, average="macro",    zero_division=0))
    f1_weighted = float(f1_score(y_test_int, y_pred, average="weighted", zero_division=0))
    mcc         = float(matthews_corrcoef(y_test_int, y_pred))
    f1_per_cls  = f1_score(y_test_int, y_pred, average=None, zero_division=0)

    per_class = {}
    for cls_idx in range(num_classes):
        mask = y_test_int == cls_idx
        if mask.sum() == 0:
            continue
        per_class[str(le.classes_[cls_idx])] = {
            "acc": float(np.mean(y_pred[mask] == cls_idx)),
            "f1":  float(f1_per_cls[cls_idx]) if cls_idx < len(f1_per_cls) else 0.0,
            "n":   int(mask.sum()),
        }
    return acc, f1_macro, f1_weighted, mcc, per_class


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    out_dir = os.path.join(PERSISTENT_PATHS["models_dir"], "finetune_runs")
    os.makedirs(out_dir, exist_ok=True)
    run_id = f"{args.experiment_name}_{int(time.time())}"

    result = {"status": "error", "message": "uninitialized"}

    try:
        hp = json.loads(args.params_json)
        # Normalize augmentation alias: builders read "use_data_augmentation", but
        # manifests/prompts use "augmentation" — map it so the flag is not a no-op.
        if "augmentation" in hp and "use_data_augmentation" not in hp:
            hp["use_data_augmentation"] = bool(hp.pop("augmentation"))
        representation_type = hp.get("representation_type")
        if not representation_type:
            raise ValueError("hyperparameters must include 'representation_type'")

        # Validate data presence
        X_train, X_test, y_train_int, y_test_int, le, num_classes = load_pretrain_data(
            args.data_path_root, representation_type)

        X_ft, y_ft_int = load_finetune_data(args.data_path_root, representation_type, le)
        if X_ft is None:
            raise FileNotFoundError(
                f"Finetune set not found for representation '{representation_type}' in {args.data_path_root}. "
                f"Expected: {representation_type}_X_finetune.npy + labels_finetune.npy. "
                f"Re-run generate_representation on a manifest that has an X_finetune.npy raw file."
            )

        model_builder = load_model_builder(
            args.custom_architecture_file, args.experiment_name, representation_type)

        # Configure GPU
        import tensorflow as tf
        for g in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(g, True)

        seeds = json.loads(args.seeds_json) if args.seeds_json else [42, 1, 7]
        print(f"[Worker] Seeds: {seeds}")

        # Phase 1
        best_pretrain_path, pretrain_best_val, pretrain_seeds = pretrain_multiseed(
            model_builder, X_train, y_train_int, X_test, y_test_int,
            num_classes, hp, seeds, out_dir, run_id)

        from tensorflow import keras
        pre_model = keras.models.load_model(best_pretrain_path)
        pretrain_ho, pretrain_f1, pretrain_f1w, pretrain_mcc, pretrain_per_class = evaluate(
            pre_model, X_test, y_test_int, le, num_classes)
        print(f"[Phase 1] Best seed HO accuracy: {pretrain_ho:.4f}  F1-macro={pretrain_f1:.4f}  MCC={pretrain_mcc:.4f}")
        keras.backend.clear_session()

        # Phase 2 — multi-seed finetune with soft-voting on shifted test.
        # Each seed trains on the FULL finetune set; the seed only varies training
        # randomness (init / shuffle) to keep the soft-voting ensemble diverse.
        ft_split_seeds = [None, 7, 99]  # None → unseeded, then two fixed alternates
        ft_kwargs = dict(
            best_pretrain_path=best_pretrain_path,
            X_ft=X_ft, y_ft_int=y_ft_int, num_classes=num_classes,
            ft_lr=float(hp.get("ft_lr", 1e-4)),
            ft_epochs=int(hp.get("ft_epochs", 200)),
            out_dir=out_dir, run_id=run_id, hp=hp,
        )

        ft_vals = []
        ft_paths = []
        for s in ft_split_seeds:
            m, v, p = finetune_full_model(**ft_kwargs, split_seed=s)
            ft_vals.append(v)
            ft_paths.append(p)
            del m
            keras.backend.clear_session()  # free GPU memory between finetune seeds

        # Reload ALL finetuned models into a SINGLE clean session for best-seed
        # evaluation and the Phase-3 ensemble. Reloading HERE — after the final
        # clear_session — is critical: keeping a model object across a *later*
        # iteration's clear_session() wipes its backend graph, so later using it
        # (best-seed evaluate / ensemble predict) crashes the process natively
        # (SIGKILL -9, no result written) whenever the best seed isn't the last one.
        ft_models = [keras.models.load_model(p) for p in ft_paths]

        # HO accuracy: pick the seed with the highest final training accuracy (single
        # model, no leakage from shifted set — selector never sees the shifted data).
        best_seed_idx = int(np.argmax(ft_vals))
        best_ft_model = ft_models[best_seed_idx]
        finetune_ho, ft_f1, ft_f1w, ft_mcc, finetune_per_class = evaluate(
            best_ft_model, X_test, y_test_int, le, num_classes)
        ft_val = ft_vals[best_seed_idx]
        ft_path = ft_paths[best_seed_idx]
        print(f"[Phase 2] Best-seed HO accuracy: {finetune_ho:.4f}  F1-macro={ft_f1:.4f}  MCC={ft_mcc:.4f}  (seed={ft_split_seeds[best_seed_idx]})")
        print(f"[Phase 2] Delta vs pretrain : {finetune_ho - pretrain_ho:+.4f}")

        # Phase 3 — shifted test: ensemble soft-vote across all ft seeds
        shifted_acc = None
        shifted_per_class = None
        shifted_acc_per_seed = []
        shifted_f1_macro = shifted_f1_weighted = shifted_mcc = None
        shifted_x_path = os.path.join(args.data_path_root, f"{representation_type}_X_shifted_test.npy")
        shifted_y_path = os.path.join(args.data_path_root, "labels_shifted_test.npy")
        if os.path.exists(shifted_x_path) and os.path.exists(shifted_y_path):
            try:
                X_shifted = np.load(shifted_x_path).astype("float32")
                y_shifted_raw = np.load(shifted_y_path, allow_pickle=True)
                y_shifted_int = le.transform(y_shifted_raw)

                all_probs = []
                for i, m in enumerate(ft_models):
                    probs = m.predict(X_shifted, batch_size=min(32, len(X_shifted)), verbose=0)
                    seed_acc = float(np.mean(probs.argmax(axis=1) == y_shifted_int))
                    shifted_acc_per_seed.append({"seed": ft_split_seeds[i], "acc": seed_acc})
                    print(f"[Phase 3] Seed {ft_split_seeds[i]} shifted acc: {seed_acc:.4f}")
                    all_probs.append(probs)

                ensemble_probs = np.mean(all_probs, axis=0)
                ensemble_preds = ensemble_probs.argmax(axis=1)
                shifted_acc = float(np.mean(ensemble_preds == y_shifted_int))

                from sklearn.metrics import f1_score as _f1, matthews_corrcoef as _mcc
                shifted_f1_macro    = float(_f1(y_shifted_int, ensemble_preds, average="macro",    zero_division=0))
                shifted_f1_weighted = float(_f1(y_shifted_int, ensemble_preds, average="weighted", zero_division=0))
                shifted_mcc         = float(_mcc(y_shifted_int, ensemble_preds))
                shifted_f1_per_cls  = _f1(y_shifted_int, ensemble_preds, average=None, zero_division=0)

                per_class = {}
                for cls_idx in range(num_classes):
                    mask = y_shifted_int == cls_idx
                    if mask.sum() == 0:
                        continue
                    per_class[str(le.classes_[cls_idx])] = {
                        "acc": float(np.mean(ensemble_preds[mask] == cls_idx)),
                        "f1":  float(shifted_f1_per_cls[cls_idx]) if cls_idx < len(shifted_f1_per_cls) else 0.0,
                        "n":   int(mask.sum()),
                    }
                shifted_per_class = per_class
                print(f"[Phase 3] Ensemble shifted acc ({len(ft_models)} seeds): {shifted_acc:.4f}  "
                      f"F1-macro={shifted_f1_macro:.4f}  MCC={shifted_mcc:.4f}")
                print(f"[Phase 3] Distribution-shift gap (HO - shifted): {finetune_ho - shifted_acc:+.4f}")
            except Exception as shift_e:
                print(f"[Phase 3] Shifted test evaluation failed: {shift_e}")
        else:
            print(f"[Phase 3] No shifted test set found at {shifted_x_path} — skipping.")

        result = {
            "status": "completed",
            "run_id": run_id,
            "seeds": seeds,
            "ft_split_seeds": ft_split_seeds,
            "representation_type": representation_type,
            "custom_architecture_file": args.custom_architecture_file,
            "experiment_name": args.experiment_name,
            "hyperparameters": hp,

            "pretrain": {
                "ho_accuracy":    float(pretrain_ho),
                "best_val_acc":   float(pretrain_best_val),
                "best_model":     best_pretrain_path,
                "seed_results":   pretrain_seeds,
                "per_class":      pretrain_per_class,
            },
            "finetune": {
                "ho_accuracy":          float(finetune_ho),
                "f1_macro":             ft_f1,
                "f1_weighted":          ft_f1w,
                "mcc":                  ft_mcc,
                "final_train_acc":      float(ft_val),
                "model_path":           ft_path,
                "all_model_paths":      ft_paths,
                "per_class":            finetune_per_class,
                "ft_unfreeze_last_n":   int(hp.get("ft_unfreeze_last_n", 0)),
                "ft_lr":                float(hp.get("ft_lr", 1e-4)),
            },
            "delta_pretrain_to_finetune": float(finetune_ho - pretrain_ho),
            "shifted_test_accuracy":      float(shifted_acc) if shifted_acc is not None else None,
            "shifted_f1_macro":           shifted_f1_macro,
            "shifted_f1_weighted":        shifted_f1_weighted,
            "shifted_mcc":                shifted_mcc,
            "shifted_test_per_class":     shifted_per_class,
            "shifted_test_per_seed":      shifted_acc_per_seed,
            "distribution_shift_gap":     float(finetune_ho - shifted_acc) if shifted_acc is not None else None,
        }

    except Exception as e:
        result = {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc(),
            "run_id": run_id,
        }
        print(f"[Worker CRASH] {e}")
        traceback.print_exc()

    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[Worker] Result written to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--params_json", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--data_path_root", required=True)
    parser.add_argument("--custom_architecture_file", default=None)
    parser.add_argument("--seeds_json", default=None, help="JSON list of seeds, default [42,1,7]")
    parser.add_argument("--gpu_id", type=int, default=None)
    args = parser.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    main(args)
