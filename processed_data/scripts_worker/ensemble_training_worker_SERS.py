# ensemble_training_worker.py
import argparse
import gc
import importlib.util
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

from state_manager import PERSISTENT_PATHS

# 1. Trova il percorso della cartella principale del progetto (due livelli sopra lo script corrente)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# 2. Aggiungi la cartella principale al percorso di ricerca di Python
sys.path.append(PROJECT_ROOT)

# 3. Ora l'importazione funzionerà correttamente

# --- CONFIGURATIONS ---


# --- FIX START: Model building functions are now included directly in this file ---


def build_1d_cnn(input_shape, num_classes, params):
    max_layers = int(np.log2(input_shape[0])) if input_shape[0] > 0 else 1
    requested_layers = params.get("num_conv_layers", 2)
    num_conv_layers = min(requested_layers, max_layers)
    if num_conv_layers < requested_layers:
        print(
            f"[Builder Warning] 1D_CNN: Requested {requested_layers} layers, but data dimension only supports {num_conv_layers}. Adjusting automatically."
        )
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for i in range(num_conv_layers):
        num_filters = min(512, params.get("filters", 32) * (2**i))
        x = keras.layers.Conv1D(num_filters, params.get("kernel_size", 3), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(params.get("dense_units", 128), activation="relu")(x)
    x = keras.layers.Dropout(params.get("dropout_rate", 0.5))(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.Adam(learning_rate=params.get("learning_rate", 1e-3))
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_2d_cnn(input_shape, num_classes, params):
    min_dim = min(input_shape[0], input_shape[1])
    max_layers = int(np.log2(min_dim)) if min_dim > 0 else 1
    requested_layers = params.get("num_conv_layers", 2)
    num_conv_layers = min(requested_layers, max_layers)
    if num_conv_layers < requested_layers:
        print(
            f"[Builder Warning] 2D_CNN: Requested {requested_layers} layers, but data dimension only supports {num_conv_layers}. Adjusting automatically."
        )
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for i in range(num_conv_layers):
        num_filters = min(512, params.get("filters", 32) * (2**i))
        x = keras.layers.Conv2D(
            num_filters, (params.get("kernel_size", 3), params.get("kernel_size", 3)), padding="same"
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(params.get("dense_units", 128), activation="relu")(x)
    x = keras.layers.Dropout(params.get("dropout_rate", 0.5))(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.Adam(learning_rate=params.get("learning_rate", 1e-3))
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_3d_cnn(input_shape, num_classes, params):
    min_dim = min(input_shape[1], input_shape[2])
    max_layers = int(np.log2(min_dim)) if min_dim > 0 else 1
    requested_layers = params.get("num_conv_layers", 2)
    num_conv_layers = min(requested_layers, max_layers)
    if num_conv_layers < requested_layers:
        print(
            f"[Builder Warning] 3D_CNN: Requested {requested_layers} layers, but data dimension only supports {num_conv_layers}. Adjusting automatically."
        )
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for i in range(num_conv_layers):
        num_filters = min(512, params.get("filters", 32) * (2**i))
        x = keras.layers.Conv3D(num_filters, kernel_size=params.get("kernel_size", 3), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    x = keras.layers.GlobalAveragePooling3D()(x)
    x = keras.layers.Dense(params.get("dense_units", 128), activation="relu")(x)
    x = keras.layers.Dropout(params.get("dropout_rate", 0.5))(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.Adam(learning_rate=params.get("learning_rate", 1e-3))
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# The MODEL_BUILDERS dictionary is now defined locally, removing the need for a fragile import.
MODEL_BUILDERS = {
    "1D_CNN": build_1d_cnn,
    "2D_SPECTROGRAM": build_2d_cnn,
    "2D_GAF": build_2d_cnn,
    "3D_VIDEO": build_3d_cnn,
    "3D_GAF_VIDEO": build_3d_cnn,
    "2D_CWT_SCALOGRAM": build_2d_cnn,
}
# --- FIX END ---


def load_data(dataset_path, model_type):
    data_model_type = model_type.replace(".py", "")
    data_file = os.path.join(dataset_path, f"{data_model_type}_data.npy")
    labels_file = os.path.join(dataset_path, "labels.npy")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Labels file not found in {dataset_path}")
    X = np.load(data_file)
    y = np.load(labels_file, allow_pickle=True)
    return X, y


def get_model_builder(architecture_name):
    if architecture_name.endswith(".py"):
        filepath = os.path.join(PERSISTENT_PATHS["custom_architectures_dir"], architecture_name)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Custom architecture file not found: {architecture_name}")
        spec = importlib.util.spec_from_file_location(architecture_name.replace(".py", ""), filepath)
        custom_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_module)
        if not hasattr(custom_module, "build_model"):
            raise AttributeError(f"'build_model' function not found in {architecture_name}")
        return custom_module.build_model
    else:
        builder = MODEL_BUILDERS.get(architecture_name)
        if builder is None:
            raise ValueError(f"Standard model builder for '{architecture_name}' not found.")
        return builder


def main(args):
    """
    Orchestrates the final training of champion models, retraining them on a
    single, consistent base dataset for fairness and reliability.
    """
    result = {}
    try:
        os.makedirs(PERSISTENT_PATHS["ensemble_models_dir"], exist_ok=True)
        print("--- Starting Final Ensemble Training ---")

        custom_config_path = os.path.join(PERSISTENT_PATHS["ensemble_models_dir"], "custom_ensemble_champions.json")

        if os.path.exists(custom_config_path):
            print("Found custom_ensemble_champions.json. Using manual configuration.")
            with open(custom_config_path, "r") as f:
                champions_list = json.load(f)
            best_champions = pd.DataFrame(champions_list)
            os.remove(custom_config_path)
        else:
            print("No custom configuration found. Finding best champions automatically from results log.")
            if not os.path.exists(PERSISTENT_PATHS["results_log"]):
                raise FileNotFoundError(
                    f"Results log not found at {PERSISTENT_PATHS["results_log"]}. Phase 1 must be completed first."
                )
            df = pd.read_json(PERSISTENT_PATHS["results_log"])
            if df.empty:
                raise ValueError("Results log is empty. No champions to train.")
            best_champions = df.loc[df.groupby("architecture")["mean_accuracy"].idxmax()]

        print(f"Using '{args.base_test_dataset_id}' as the single source for all training data.")
        base_dataset_path = os.path.join(PERSISTENT_PATHS["processed_data_dir"], args.base_test_dataset_id)

        labels_path = os.path.join(base_dataset_path, "labels.npy")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found in base dataset: {labels_path}")
        y_base = np.load(labels_path, allow_pickle=True)

        indices = np.arange(len(y_base))
        train_indices, test_indices = train_test_split(
            indices, test_size=args.test_split_ratio, random_state=42, stratify=y_base
        )

        test_indices_path = os.path.join(PERSISTENT_PATHS["ensemble_models_dir"], "holdout_test_indices.npy")
        train_indices_path = os.path.join(PERSISTENT_PATHS["ensemble_models_dir"], "holdout_train_indices.npy")
        np.save(test_indices_path, test_indices)
        np.save(train_indices_path, train_indices)

        le = LabelEncoder()
        le.fit(y_base)
        np.save(os.path.join(PERSISTENT_PATHS["ensemble_models_dir"], "label_encoder_classes.npy"), le.classes_)
        num_classes = len(le.classes_)

        print("\nChampions to be retrained:")
        display_cols = [
            "architecture",
            "manifest_name" if "manifest_name" in best_champions.columns else "dataset_id",
            "mean_accuracy",
        ]
        print(best_champions[display_cols].to_string())

        trained_models_paths = []
        for _, champion in best_champions.iterrows():
            arch_name = champion["architecture"]
            params = champion["params"]

            print(f"\n--- Training champion: {arch_name} ---")
            print(f"Using best hyperparameters on consistent base dataset '{args.base_test_dataset_id}'")

            try:
                X_full, y_full = load_data(base_dataset_path, arch_name)

                if len(X_full) != len(y_base):
                    raise ValueError(
                        f"Data mismatch in manifest '{args.base_test_dataset_id}'! Representation '{arch_name}' has {len(X_full)} samples, but labels file has {len(y_base)}. Please regenerate the dataset."
                    )

                X_train, y_train_labels = X_full[train_indices], y_full[train_indices]

                y_train_int = le.transform(y_train_labels)
                y_train_cat = keras.utils.to_categorical(y_train_int, num_classes=num_classes)

                model_builder = get_model_builder(arch_name)
                model = model_builder(X_train.shape[1:], num_classes, params)

                print(f"Training on data with shape: {X_train.shape}")
                model.fit(
                    X_train,
                    y_train_cat,
                    epochs=150,
                    batch_size=params.get("batch_size", 16),
                    validation_split=0.15,
                    callbacks=[
                        keras.callbacks.EarlyStopping(patience=15, monitor="val_loss", restore_best_weights=True)
                    ],
                    verbose=1,
                )

                model_path = os.path.join(
                    PERSISTENT_PATHS["ensemble_models_dir"], f"champion_{arch_name.replace('.py', '')}.keras"
                )
                model.save(model_path)
                trained_models_paths.append(model_path)
                print(f"✅ Champion '{arch_name}' trained and saved to {model_path}")

                del model, X_train, y_train_labels, y_train_cat, X_full, y_full
                gc.collect()
                keras.backend.clear_session()

            except Exception as e:
                print(f"❌ FAILED to train champion {arch_name}. Error: {e}")
                traceback.print_exc()

        result = {
            "status": "completed",
            "message": f"{len(trained_models_paths)} champion models trained.",
            "model_paths": trained_models_paths,
        }

    except Exception as e:
        error_trace = traceback.format_exc()
        result = {"status": "error", "message": str(e), "traceback": error_trace}

    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Worker for training final ensemble models.")
    parser.add_argument(
        "--base_test_dataset_id",
        type=str,
        required=True,
        help="Dataset ID to use for creating the final holdout test set.",
    )
    parser.add_argument(
        "--test_split_ratio", type=float, default=0.2, help="Fraction of the data to hold out for final testing."
    )
    parser.add_argument("--output_path", type=str, required=True, help="Path to write the JSON result file.")

    parsed_args = parser.parse_args()
    main(parsed_args)
