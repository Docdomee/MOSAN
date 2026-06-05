# reinforcement_learner.py
import os
from typing import List

import keras
import numpy as np
import tensorflow as tf
from keras import layers

from ui_logger import log


class MemoryRanker:
    """
    A simple neural network that learns to rank the usefulness of a memory
    in a given context (query) using Q-learning.
    """

    def __init__(self, embedding_dim: int, model_path_dir: str):
        """
        Initializes the MemoryRanker.
        Args:
            embedding_dim (int): The dimension of the text embeddings.
            model_path_dir (str): The directory to save/load the model weights.
        """
        self.embedding_dim = embedding_dim
        self.model_path_dir = model_path_dir  # Ora riceve il percorso corretto da chi lo crea
        os.makedirs(self.model_path_dir, exist_ok=True)

        self.optimizer = keras.optimizers.Adam(learning_rate=1e-4)
        self.loss_fn = keras.losses.Huber()
        self.model = self._build_model()

    def _get_model_path(self) -> str:
        """Constructs the file path for the global model weights."""
        return os.path.join(self.model_path_dir, "global_ranker.weights.h5")

    def load_weights(self):
        """Loads the model weights if they exist."""
        model_path = self._get_model_path()
        if os.path.exists(model_path):
            try:
                self.model.load_weights(model_path)
                log(f"[RL MemoryRanker] Global weights loaded successfully from {model_path}")
            except Exception as e:
                log(f"[RL MemoryRanker] ERROR: Could not load weights. Starting with an untrained model. Error: {e}")
        else:
            log("[RL MemoryRanker] No pre-trained weights found. Starting with a new model.")

    def save_weights(self):
        """Saves the model weights."""
        model_path = self._get_model_path()
        self.model.save_weights(model_path)
        log(f"[RL MemoryRanker] Global weights saved successfully to {model_path}")

    def _build_model(self) -> keras.Model:
        """Builds the Keras model for the Q-network."""
        query_input = keras.Input(shape=(self.embedding_dim,), name="query_input")
        memory_input = keras.Input(shape=(self.embedding_dim,), name="memory_input")

        concatenated = layers.concatenate([query_input, memory_input])

        x = layers.Dense(128, activation="relu")(concatenated)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation="relu")(x)

        output_tensor = layers.Dense(1, activation="linear", name="q_value")(x)

        model = keras.Model(inputs=[query_input, memory_input], outputs=output_tensor)
        model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        return model

    def predict_value(self, query_embedding: np.ndarray, memory_embeddings: List[np.ndarray]) -> np.ndarray:
        """Predicts the Q-value for one or more memories."""
        num_memories = len(memory_embeddings)
        query_batch = np.repeat(query_embedding[np.newaxis, :], num_memories, axis=0)
        memory_batch = np.array(memory_embeddings)

        q_values = self.model.predict([query_batch, memory_batch], verbose=0)
        return q_values.flatten()

    def train_step(self, experiences: List[tuple], gamma: float = 0.9):
        """Performs a single training step on a batch of experiences."""
        if not experiences:
            return 0.0

        query_embeddings, memory_embeddings, rewards = zip(*experiences)

        query_batch = np.array(query_embeddings)
        memory_batch = np.array(memory_embeddings)
        rewards_batch = np.array(rewards)

        target_q_values = rewards_batch

        with tf.GradientTape() as tape:
            predicted_q_values = self.model([query_batch, memory_batch], training=True)
            loss = self.loss_fn(target_q_values, predicted_q_values)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss
