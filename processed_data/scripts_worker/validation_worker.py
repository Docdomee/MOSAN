# validation_worker.py

import argparse
import gc
import importlib.util
import json
import os
import sys
import traceback

from tensorflow.keras import backend as K

# 1. Trova il percorso della cartella principale del progetto (due livelli sopra lo script corrente)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# 2. Aggiungi la cartella principale al percorso di ricerca di Python
sys.path.append(PROJECT_ROOT)

# 3. Ora l'importazione funzionerà correttamente
from state_manager import PERSISTENT_PATHS


def main(filename, output_path, expected_input_type):
    filepath = os.path.join(PERSISTENT_PATHS["custom_architectures_dir"], filename)

    try:
        # --- FIX 2: Auto-Generate Wrapper if Missing ---
        if not os.path.exists(filepath):
            print(f"[Validation] File {filename} not found. Attempting to auto-generate from model_factory...")
            try:
                # Try to map filename to MODEL_BUILDERS key
                # Accepted formats: "1D_CNN_baseline.py" -> "1D_CNN", "2D_GAF.py" -> "2D_GAF"
                potential_key = filename.replace("_baseline.py", "").replace(".py", "")
                
                # Import model_factory dynamically to check keys
                from model_factory import MODEL_BUILDERS
                
                if potential_key in MODEL_BUILDERS:
                    print(f"[Validation] Found '{potential_key}' in MODEL_BUILDERS. Creating wrapper.", flush=True)
                    try:
                        print(f"[Validation] Opening file for write: {filepath}", flush=True)
                        with open(filepath, "w") as f:
                            print(f"[Validation] Writing imports...", flush=True)
                            f.write(f"from model_factory import MODEL_BUILDERS\n\n")
                            print(f"[Validation] Writing function definition...", flush=True)
                            f.write(f"def build_model(input_shape, num_classes, params):\n")
                            f.write(f"    # Auto-generated wrapper for {potential_key}\n")
                            f.write(f"    builder = MODEL_BUILDERS['{potential_key}']\n")
                            f.write(f"    return builder(input_shape, num_classes, params)\n")
                        print(f"[Validation] Wrapper created successfully.", flush=True)
                    except Exception as write_err:
                        print(f"[Validation] ERROR writing file: {write_err}", flush=True)
                        raise
                else:
                    print(f"[Validation] '{potential_key}' not found in MODEL_BUILDERS. Available: {list(MODEL_BUILDERS.keys())}", flush=True)
            except Exception as e:
                print(f"[Validation] Auto-generation failed: {e}", flush=True)
                traceback.print_exc()

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File di architettura '{filename}' non trovato e generazione fallita.")

        spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), filepath)
        if spec is None or spec.loader is None:
            raise ImportError(f"Impossibile creare lo spec per '{filename}'.")

        custom_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_module)

        if not hasattr(custom_module, "build_model"):
            raise AttributeError("Il file di architettura non ha una funzione 'build_model'.")

        # --- Logica Chiave: Crea il dummy_shape corretto ---
        if expected_input_type == "1D":
            dummy_shape = (128, 1)
        elif expected_input_type == "3D":
             dummy_shape = (10, 64, 64, 1)
        else:
             dummy_shape = (64, 64, 1)

        # --- FIX 1: SmartDefaultDict for Robust Params ---
        class SmartDefaultDict(dict):
            def __getitem__(self, key):
                if key not in self:
                    # Heuristic defaults for common hyperparameters
                    if 'rate' in key: val = 0.5
                    elif 'size' in key: val = 3
                    elif 'units' in key or 'filters' in key: val = 32
                    elif 'layers' in key or 'blocks' in key: val = 1
                    elif 'gamma' in key or 'lambda' in key: val = 0.01
                    else: val = 1 # Fallback generic
                    
                    print(f"[Validation Warning] Missing param '{key}', using default: {val}")
                    return val
                return super().__getitem__(key)

        base_params = {
            "learning_rate": 1e-3,
            "dropout_rate": 0.5,
            # LIST-BASED DEFAULTS
            "filters_list": [32, 64, 128],
            "kernel_size_list": [3, 3, 3],
            "stride_list": [1, 1, 1],
            "num_blocks": 3,
            # SINGLE VALUES
            "filters": 32,
            "kernel_size": 3,
            "stride": 1,
            "initial_filters": 32,
            "initial_kernel_size": 3,
            "initial_stride": 1,
            "dense_units": 64,
            "num_conv_layers": 2,
            "batch_size": 32,
            "num_classes": 10,
            # REGULARIZATION
            "regularizer": "l2",
            "l1_rate": 0.01,
            "l2_rate": 0.01,
            "model_name": "validated_model"
        }
        dummy_params = SmartDefaultDict(base_params)

        # Costruisce il modello per verificare che non ci siano errori
        model = custom_module.build_model(input_shape=dummy_shape, num_classes=10, params=dummy_params)

        result = {"status": "completed", "message": "Validazione completata con successo."}

    except Exception:
        result = {
            "status": "error",
            "message": "Validazione fallita. Analizza il traceback.",
            "traceback": traceback.format_exc(),
        }

    finally:
        # Pulizia della memoria
        if "model" in locals():
            del model
        K.clear_session()
        gc.collect()

    with open(output_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--expected_input_type", required=True)  # <-- Nuovo argomento
    args = parser.parse_args()
    main(args.filename, args.output_path, args.expected_input_type)
