# Changelog for regularized_volumetric_cwt_resnet.py

## Version 1.0 (Attempt 1)
- Initial creation based on hypothesis: Improve dynamic_cwt_volumetric_f_baseline.py using advanced techniques.



## PASSED VALIDATION (Attempt 1)
- Code is syntactically valid. Proceeding to training trial.



## FAILED TRAINING (Attempt 1)
**Error:**
```
Training Worker Failure Details:
Training error in Fold 1: Traceback (most recent call last):
  File "/mnt/dati/homes/dsagnelli/Agent_super_celery/cluster_upgrade/processed_data/scripts_worker/training_worker.py", line 556, in main
    model = model_builder(X_train.shape[1:], num_classes, hyperparameters)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/dati/homes/dsagnelli/Agent_super_celery/cluster_upgrade/processed_data/custom_architectures/regularized_volumetric_cwt_resnet.py", line 46, in build_model
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/dati/homes/dsagnelli/miniconda3/envs/agent_gpu_env/lib/python3.11/site-packages/keras/src/utils/traceback_utils.py", line 70, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "/mnt/dati/homes/dsagnelli/miniconda3/envs/agent_gpu_env/lib/python3.11/site-packages/tensorflow/python/framework/ops.py", line 1020, in _create_c_op
    raise ValueError(e.message)
ValueError: Exception encountered when calling layer "max_pooling3d_6" (type MaxPooling3D).

Negative dimension size caused by subtracting 2 from 1 for '{{node max_pooling3d_6/MaxPool3D}} = MaxPool3D[T=DT_HALF, data_format="NDHWC", ksize=[1, 2, 2, 2, 1], padding="VALID", strides=[1, 2, 2, 2, 1]](Placeholder)' with input shapes: [?,1,3,3,512].

Call arguments received by layer "max_pooling3d_6" (type MaxPooling3D):
  • inputs=tf.Tensor(shape=(None, 1, 3, 3, 512), dtype=float16)


WORKER LOG:
```



## DEBUG FIX APPLIED (Attempt 1)
- Debugger applied a new fix for runtime error. Re-validating...



## PASSED VALIDATION (Attempt 2)
- Code is syntactically valid. Proceeding to training trial.



## SUCCESS (Attempt 2)
- Architecture passed validation and training.
- **Final Result:** {"status": "completed", "mean_accuracy": 0.5017701288064321, "mean_train_accuracy": 0.8688274025917053, "mean_heldout_accuracy": 0.49319212635358173, "held_out_test_accuracy": 0.7397881746292114, "shifted_test_accuracy": 0.34455129504203796, "evaluation_error": null, "std_accuracy": 0.03939595332991456, "mean_final_val_loss": 1.5295019149780273, "mean_training_epochs": 7.333333333333333, "mean_overfitting_score": 0.3785417675971985, "mean_learning_speed": 0.014659061937621154, "mean_convergence_stability": 0.09505685715833963, "model_path": "Model not saved (save_model=False)", "effective_hyperparameters": {"num_conv_layers": 2, "num_layers": 2, "filters": 32, "kernel_size": 3, "dense_units": 128, "dropout_rate": 0.5, "learning_rate": 0.001, "batch_size": 32, "epochs": 10, "patience": 5, "batch_norm": 0.0, "weight_decay": 0.0, "use_residual": false, "use_l1_regularization": false, "use_l2_regularization": false, "use_data_augmentation": false, "representation_type": "3D_DYNAMIC_GAF", "save_model": false, "checkpoint_every_n_epochs": 20}, "checkpoint_history": [], "inference_time_ms": 19.478096961975098, "params_count": 54726659, "model_size_mb": 208.7656364440918, "architecture": "regularized_volumetric_cwt_resnet.py", "manifest_name": "SERS_data_test", "params": {"num_conv_layers": 2, "num_layers": 2, "filters": 32, "kernel_size": 3, "dense_units": 128, "dropout_rate": 0.5, "learning_rate": 0.001, "batch_size": 32, "epochs": 10, "patience": 5, "batch_norm": 0.0, "weight_decay": 0.0, "use_residual": false, "use_l1_regularization": false, "use_l2_regularization": false, "use_data_augmentation": false, "representation_type": "3D_DYNAMIC_GAF", "save_model": false, "checkpoint_every_n_epochs": 20}, "source": "agent_trial", "missing_params_warning": ["learning_rate", "dropout_rate", "filters", "kernel_size", "batch_size", "weight_decay", "batch_norm"]}

