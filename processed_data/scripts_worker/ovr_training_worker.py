# ovr_training_worker.py

import argparse
import gc
import json
import os
import sys
import traceback

import numpy as np
from ensemble_training_worker import load_data
from tensorflow import keras

from state_manager import PERSISTENT_PATHS

# 1. Trova il percorso della cartella principale del progetto (due livelli sopra lo script corrente)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# 2. Aggiungi la cartella principale al percorso di ricerca di Python
sys.path.append(PROJECT_ROOT)

# 3. Ora l'importazione funzionerà correttamente


def main(args):
    """
    Main function to train a single OvR specialist model using class weights.
    """
    result = {}
    model = None
    try:
        print(f"--- Starting OvR Fine-Tuning for Specialist: {args.target_class} with Class Weights ---")

        # 1. Load data and indices
        champion_dataset_path = os.path.join(PERSISTENT_PATHS["processed_data_dir"], args.dataset_id)
        X_champ, y_champ_multiclass = load_data(champion_dataset_path, args.architecture_name)
        train_indices_path = os.path.join(PERSISTENT_PATHS["ensemble_models_dir"], "holdout_train_indices.npy")
        train_indices = np.load(train_indices_path)
        X_train = X_champ[train_indices]
        y_train_multiclass = y_champ_multiclass[train_indices]

        # 2. Transform labels to binary
        y_train_binary = np.array([1 if label == args.target_class else 0 for label in y_train_multiclass])

        # --- Use Class Weights instead of Undersampling ---
        print(f"Original training data shape: {X_train.shape}")
        print("Calculating class weights for imbalanced dataset...")

        num_positives = np.sum(y_train_binary == 1)
        num_negatives = np.sum(y_train_binary == 0)
        total_samples = len(y_train_binary)

        if num_positives == 0 or num_negatives == 0:
            raise ValueError(
                f"Cannot train OvR specialist for class {args.target_class}: only one class present in training data."
            )

        weight_for_0 = (1 / num_negatives) * (total_samples / 2.0)
        weight_for_1 = (1 / num_positives) * (total_samples / 2.0)
        class_weights = {0: weight_for_0, 1: weight_for_1}

        print(f"Class weights: {class_weights}")
        # --- End Class Weights ---

        # 3. Carica il modello campione già addestrato (logica di fine-tuning)
        arch_name_clean = args.architecture_name.replace(".py", "")
        champion_model_path = os.path.join(
            PERSISTENT_PATHS["ensemble_models_dir"], f"champion_{arch_name_clean}.keras"
        )
        base_model = keras.models.load_model(champion_model_path)

        # 4. Congela i suoi layer per preservare le feature, ma scongela gli ultimi N
        for i, layer in enumerate(base_model.layers):
            if i < len(base_model.layers) - args.fine_tune_layers:
                layer.trainable = False
            else:
                layer.trainable = True

        # 5. Crea la nuova testa binaria e il modello specialista
        # Rimuovi l'ultima layer (output layer) del modello base
        x = base_model.layers[-2].output
        binary_output = keras.layers.Dense(1, activation="sigmoid", name="binary_output")(x)
        model = keras.Model(inputs=base_model.input, outputs=binary_output)

        # 6. Compila il modello
        params = json.loads(args.hyperparameters)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params.get("learning_rate", 5e-4)),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        # 7. Addestra il modello con class_weights
        print(f"Fine-tuning specialist on data with shape: {X_train.shape}")
        model.fit(
            X_train,
            y_train_binary,
            epochs=args.epochs,  # Use configurable epochs
            batch_size=params.get("batch_size", 32),
            validation_split=0.15,
            callbacks=[keras.callbacks.EarlyStopping(patience=15, monitor="val_loss", restore_best_weights=True)],
            class_weight=class_weights,  # Apply class weights
            verbose=1,
        )

        model_dir = os.path.join(PERSISTENT_PATHS["ensemble_models_dir"], f"ovr_{arch_name_clean}")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"specialist_{args.target_class}.keras")
        model.save(model_path)
        print(f"✅ OvR Specialist '{args.target_class}' for '{arch_name_clean}' trained and saved to {model_path}")
        result = {"status": "completed", "model_path": model_path}

    except Exception as e:
        error_trace = traceback.format_exc()
        result = {"status": "error", "message": str(e), "traceback": error_trace}
        print(f"❌ FAILED to train OvR specialist for {args.target_class}. Error: {e}")

    finally:
        with open(args.output_path, "w") as f:
            json.dump(result, f, indent=2)
        if model is not None:
            del model
        if "base_model" in locals():
            del base_model
        gc.collect()
        keras.backend.clear_session()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Worker for training a single OvR specialist model.")
    parser.add_argument("--architecture_name", type=str, required=True)
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--hyperparameters", type=str, required=True)
    parser.add_argument("--target_class", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs for training.")
    parser.add_argument(
        "--fine_tune_layers", type=int, default=0, help="Number of top layers to fine-tune in the base model."
    )
    parser.add_argument("--output_path", type=str, required=True, help="Path to write the JSON result file.")
    parsed_args = parser.parse_args()
    main(parsed_args)
