# ovr_evaluation_worker.py
import argparse
import gc
import json
import os
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ensemble_evaluation_worker import load_data
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

from state_manager import PERSISTENT_PATHS

# 1. Trova il percorso della cartella principale del progetto (due livelli sopra lo script corrente)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# 2. Aggiungi la cartella principale al percorso di ricerca di Python
sys.path.append(PROJECT_ROOT)

# 3. Ora l'importazione funzionerà correttamente


def main(args):
    """
    Main function to evaluate the ensemble of One-vs-Rest specialist models.
    """
    result = {}
    try:
        print("--- Starting Final One-vs-Rest (OvR) Ensemble Evaluation ---")

        # 1. Load setup files (correct)
        test_indices_path = os.path.join(PERSISTENT_PATHS["ensemble_models_dir"], "holdout_test_indices.npy")
        le_classes_path = os.path.join(PERSISTENT_PATHS["ensemble_models_dir"], "label_encoder_classes.npy")

        if not (os.path.exists(test_indices_path) and os.path.exists(le_classes_path)):
            raise FileNotFoundError("Required files (indices or encoder) not found.")

        test_indices = np.load(test_indices_path)
        le_classes = np.load(le_classes_path, allow_pickle=True)
        le = LabelEncoder()
        le.classes_ = le_classes
        num_classes = len(le.classes_)

        # 2. Load test data (correct)
        base_arch_name = args.architecture_name.replace(".py", "")
        dataset_path = os.path.join(PERSISTENT_PATHS["processed_data_dir"], args.dataset_id)
        X_full = load_data(dataset_path, args.architecture_name)
        X_test = X_full[test_indices]

        labels_path = os.path.join(dataset_path, "labels.npy")
        y_full = np.load(labels_path, allow_pickle=True)
        y_true_labels = y_full[test_indices]
        y_true_int = le.transform(y_true_labels)

        # --- NEW: Evaluate the original base champion model for comparison ---
        base_model_accuracy = 0
        base_model_path = os.path.join(PERSISTENT_PATHS["ensemble_models_dir"], f"champion_{base_arch_name}.keras")
        if os.path.exists(base_model_path):
            print(f"Loading original champion model for baseline comparison: {base_model_path}")
            base_model = keras.models.load_model(base_model_path)
            base_preds_proba = base_model.predict(X_test)
            base_preds_int = np.argmax(base_preds_proba, axis=1)
            base_model_accuracy = accuracy_score(y_true_int, base_preds_int)
            print(f"Baseline accuracy of original model: {base_model_accuracy:.4f}")
            del base_model
            gc.collect()
            keras.backend.clear_session()
        else:
            print(f"WARNING: Could not find original champion model at {base_model_path} for comparison.")
        # --- END NEW ---

        # 3. Find and evaluate the OvR specialists
        model_dir = os.path.join(PERSISTENT_PATHS["ensemble_models_dir"], f"ovr_{base_arch_name}")
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Could not find the model directory for OvR specialists: {model_dir}")

        # Collect all specialist models and their target classes
        specialist_models = {}
        for class_name in le.classes_:
            model_file = f"specialist_{class_name}.keras"
            model_path = os.path.join(model_dir, model_file)
            if os.path.exists(model_path):
                specialist_models[class_name] = model_path
            else:
                print(
                    f"WARNING: Specialist model for class '{class_name}' not found at {model_path}. Skipping this class."
                )

        if len(specialist_models) != num_classes:
            raise ValueError(
                f"Expected {num_classes} specialist models, but found {len(specialist_models)}. Ensure all classes have a trained specialist."
            )

        print(f"Found {len(specialist_models)} OvR specialist models to evaluate.")

        # 4. Collect predictions from each specialist, ensuring order matches le.classes_
        all_scores = []
        for class_name in le.classes_:  # Iterate in the order of LabelEncoder classes
            model_path = specialist_models[class_name]
            print(f"Loading specialist for class: {class_name} from {model_path}")
            model = keras.models.load_model(model_path)
            scores = model.predict(X_test).flatten()
            all_scores.append(scores)
            del model
            gc.collect()
            keras.backend.clear_session()

        # 5. Determine the final OvR prediction
        scores_matrix = np.array(all_scores).T  # Transpose to have (num_samples, num_classes)
        predicted_indices = np.argmax(scores_matrix, axis=1)
        predicted_int = predicted_indices  # Since indices now directly correspond to le.classes_ integer labels

        # 6. Calculate performance and compare
        ensemble_accuracy = accuracy_score(y_true_int, predicted_int)
        performance_lift = ensemble_accuracy - base_model_accuracy

        # 7. Generate report
        cm = confusion_matrix(y_true_int, predicted_int, labels=np.arange(num_classes))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f"OvR Ensemble Confusion Matrix for {base_arch_name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        cm_path = os.path.join(
            PERSISTENT_PATHS["ensemble_models_dir"], f"ensemble_confusion_matrix_OvR_{base_arch_name}.png"
        )
        plt.savefig(cm_path)

        # --- UPDATE THE REPORT ---
        report = {
            "ensemble_accuracy": ensemble_accuracy,
            "baseline_model_accuracy": base_model_accuracy,
            "performance_lift": performance_lift,
            "confusion_matrix_path": cm_path,
            "evaluated_specialists": list(le.classes_),  # Report classes in correct order
        }
        result = {"status": "completed", "message": "OvR Ensemble evaluation finished.", "report": report}

    except Exception as e:
        error_trace = traceback.format_exc()
        result = {"status": "error", "message": str(e), "traceback": error_trace}

    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Worker for evaluating the final OvR ensemble.")
    parser.add_argument(
        "--architecture_name", type=str, required=True, help="The base architecture of the specialists."
    )
    parser.add_argument(
        "--dataset_id", type=str, required=True, help="The dataset ID used for training the champions."
    )
    parser.add_argument("--output_path", type=str, required=True, help="Path to write the JSON result file.")
    parsed_args = parser.parse_args()
    main(parsed_args)
