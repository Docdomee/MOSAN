# ensemble_evaluation_worker.py
import argparse
import gc
import json
import os
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

# 1. Trova il percorso della cartella principale del progetto (due livelli sopra lo script corrente)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# 2. Aggiungi la cartella principale al percorso di ricerca di Python
sys.path.append(PROJECT_ROOT)

# 3. Ora l'importazione funzionerà correttamente
from state_manager import PERSISTENT_PATHS
print(f"DEBUG: PROJECT_ROOT calculated as: {PROJECT_ROOT}")
print(f"DEBUG: sys.path: {sys.path}")

# 3. Ora l'importazione funzionerà correttamente

# --- CONFIGURAZIONI ---


def load_data(dataset_path, model_type):
    """Carica i dati specifici per un tipo di modello da un dataset."""
    data_model_type = model_type.replace(".py", "")
    data_file = os.path.join(dataset_path, f"{data_model_type}_data.npy")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    X = np.load(data_file)
    return X


# In ensemble_evaluation_worker.py


def main(args):
    """
    Orchestrates the evaluation of the standard ensemble, including a comparison
    against the best individual model in the group.
    """
    result = {}
    try:
        print("--- Starting Final Ensemble Evaluation ---")

        test_indices_path = os.path.join(PERSISTENT_PATHS["ensemble_models_dir"], "holdout_test_indices.npy")
        le_classes_path = os.path.join(PERSISTENT_PATHS["ensemble_models_dir"], "label_encoder_classes.npy")

        if not os.path.exists(test_indices_path) or not os.path.exists(le_classes_path):
            raise FileNotFoundError("Required files from training phase (indices or encoder) not found.")

        test_indices = np.load(test_indices_path)
        le_classes = np.load(le_classes_path, allow_pickle=True)
        le = LabelEncoder()
        le.classes_ = le_classes

        model_files = [
            f
            for f in os.listdir(PERSISTENT_PATHS["ensemble_models_dir"])
            if f.startswith("champion_") and f.endswith(".keras")
        ]
        if not model_files:
            raise FileNotFoundError("No trained champion models found in the ensemble directory.")
        print(f"Found {len(model_files)} champion models to evaluate.")

        df_results = pd.read_json(PERSISTENT_PATHS["results_log"])

        # Use the passed dataset_id directly
        dataset_id = args.dataset_id
        dataset_path = os.path.join(PERSISTENT_PATHS["processed_data_dir"], dataset_id)

        labels_path = os.path.join(dataset_path, "labels.npy")
        y_full = np.load(labels_path, allow_pickle=True)
        y_true_labels = y_full[test_indices]
        y_true_int = le.transform(y_true_labels)

        all_proba_predictions = []
        individual_accuracies = {}

        # Collect predictions and accuracies for each model
        for model_file in model_files:
            model_path = os.path.join(PERSISTENT_PATHS["ensemble_models_dir"], model_file)

            # Extract architecture name and original trial index from the model filename
            model_name_parts = model_file.replace("champion_", "").replace(".keras", "").rsplit("_", 1)
            arch_name_base = model_name_parts[0]
            original_trial_idx = model_name_parts[1] if len(model_name_parts) > 1 else None

            # Reconstruct arch_name for lookup in results_log (handle .py suffix)
            arch_name_lookup = arch_name_base
            if not any(
                arch_name_base == std_arch
                for std_arch in ["1D_CNN", "2D_GAF", "2D_SPECTROGRAM", "3D_VIDEO", "3D_GAF_VIDEO", "2D_CWT_SCALOGRAM"]
            ):
                arch_name_lookup += ".py"

            champion_info = df_results[
                (df_results["architecture"] == arch_name_lookup) & (df_results.index == int(original_trial_idx))
            ]

            if champion_info.empty:
                print(
                    f"WARNING: Could not find champion info for {arch_name_lookup} (Trial Index: {original_trial_idx}) in results log. Skipping."
                )
                continue

            print(f"\nEvaluating model: {model_file} on data from dataset: {dataset_id}")

            try:
                X_full = load_data(dataset_path, arch_name_lookup)
                X_test = X_full[test_indices]

                model = keras.models.load_model(model_path)
                predictions_proba = model.predict(X_test)

                # Calculate individual accuracy for weighting
                predictions_int = np.argmax(predictions_proba, axis=1)
                acc = accuracy_score(y_true_int, predictions_int)
                individual_accuracies[model_file] = acc

                all_proba_predictions.append(predictions_proba)

                del model, X_full, X_test
                gc.collect()
                keras.backend.clear_session()
            except Exception as e:
                print(f"ERROR evaluating {model_file}: {e}")
                traceback.print_exc()

        if not all_proba_predictions:
            raise ValueError("No models were successfully evaluated.")

        # --- Weighted Averaging Ensemble ---
        # Weights based on individual model accuracy
        model_weights = np.array(list(individual_accuracies.values()))
        # Normalize weights to sum to 1
        model_weights = model_weights / np.sum(model_weights)

        # Combine probabilities using weighted average
        weighted_avg_proba = np.average(all_proba_predictions, axis=0, weights=model_weights)
        ensemble_preds_int = np.argmax(weighted_avg_proba, axis=1)

        ensemble_accuracy = accuracy_score(y_true_int, ensemble_preds_int)

        # Find the best individual model for comparison
        best_individual_accuracy = 0
        best_individual_model_name = "N/A"
        if individual_accuracies:
            best_individual_model_file = max(individual_accuracies, key=individual_accuracies.get)
            best_individual_accuracy = individual_accuracies[best_individual_model_file]
            best_individual_model_name = best_individual_model_file.replace("champion_", "").replace(".keras", "")

        performance_lift = ensemble_accuracy - best_individual_accuracy

        cm = confusion_matrix(y_true_int, ensemble_preds_int, labels=np.arange(len(le.classes_)))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title("Ensemble Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        cm_path = os.path.join(PERSISTENT_PATHS["ensemble_models_dir"], "ensemble_confusion_matrix.png")
        plt.savefig(cm_path)

        report = {
            "ensemble_accuracy": ensemble_accuracy,
            "individual_accuracies": individual_accuracies,
            "best_individual_model": {"name": best_individual_model_name, "accuracy": best_individual_accuracy},
            "performance_lift": performance_lift,
            "confusion_matrix_path": cm_path,
            "evaluated_models": list(individual_accuracies.keys()),
        }
        result = {"status": "completed", "message": "Ensemble evaluation finished.", "report": report}

    except Exception as e:
        error_trace = traceback.format_exc()
        result = {"status": "error", "message": str(e), "traceback": error_trace}

    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Worker for evaluating the final ensemble.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to write the JSON result file.")
    parser.add_argument("--dataset_id", type=str, required=True, help="The ID of the dataset to use for evaluation.")
    parsed_args = parser.parse_args()
    main(parsed_args)
