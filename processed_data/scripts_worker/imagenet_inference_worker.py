"""
processed_data/scripts_worker/imagenet_inference_worker.py

Subprocess worker for ILSVRC ImageNet inference + submission generation.
Launched by cluster.tasks.run_imagenet_inference_task via Celery.

For each test image:
    - Runs the model forward pass
    - Takes the top-5 class predictions
    - Converts predicted bbox [0,1] normalized coords → pixel integers
    - Writes Kaggle-format submission CSV
"""
import argparse
import json
import logging
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf

logging.basicConfig(level=logging.INFO, format="[%(levelname)s %(asctime)s] %(message)s")
log = logging.getLogger(__name__)

# Number of top-class predictions to generate boxes for (Kaggle max = 5)
TOP_K = 5
# Output image size used during training (bbox coords are relative to this)
IMAGE_SIZE = 224


def run_inference(args: dict, output_path: str):
    checkpoint_path = args["checkpoint_path"]
    test_dir = args["test_dir"]
    synset_map_path = args["synset_map"]
    output_dir = args.get("output_path", "outputs/imagenet")
    batch_size = args.get("batch_size", 64)
    submission_csv = os.path.join(output_dir, "submission.csv")
    predictions_json = os.path.join(output_dir, "raw_predictions.json")

    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Load synset mapping ---
    from imagenet.synset_utils import load_synset_mapping
    synset_to_int = load_synset_mapping(synset_map_path)
    int_to_label = {v: k for k, v in synset_to_int.items()}  # Not used directly, labels are ints
    num_classes = len(synset_to_int)

    # --- 2. Load model ---
    from model_factory import get_imagenet_builder
    backbone_depth = args.get("backbone_depth", 152)
    build_fn = get_imagenet_builder()
    model = build_fn(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        num_classes=num_classes,
        params={"backbone_depth": backbone_depth},
    )
    model.load_weights(checkpoint_path)
    log.info(f"[Inference] Loaded checkpoint: {checkpoint_path}")

    # --- 3. Build inference dataset ---
    from imagenet.data_loader import ImageNetDataLoader
    loader = ImageNetDataLoader(synset_to_int)
    dataset, image_ids = loader.build_inference_dataset(test_dir, batch_size)
    log.info(f"[Inference] {len(image_ids)} test images found")

    # --- 4. Run inference ---
    all_predictions = []
    processed = 0

    for batch_images, batch_orig_shapes in dataset:
        cls_preds, bbox_preds = model(batch_images, training=False)
        batch_size_actual = tf.shape(batch_images)[0].numpy()

        # Top-K class probabilities
        top_k_probs, top_k_labels = tf.math.top_k(cls_preds, k=TOP_K)
        top_k_labels = top_k_labels.numpy()  # [B, K]
        bbox_preds_np = bbox_preds.numpy()   # [B, 4] normalized [0,1]
        batch_orig_shapes_np = batch_orig_shapes.numpy() # [B, 2]

        for i in range(batch_size_actual):
            image_id = image_ids[processed + i]
            # Convert normalized bbox to integer pixel coords (relative to true image size)
            x1, y1, x2, y2 = bbox_preds_np[i]
            orig_h, orig_w = batch_orig_shapes_np[i]

            px1 = max(0, int(x1 * orig_w))
            py1 = max(0, int(y1 * orig_h))
            px2 = min(int(orig_w), int(x2 * orig_w))
            py2 = min(int(orig_h), int(y2 * orig_h))

            # Generate one prediction per top-K class, all sharing the same bbox
            # (single-bbox regression head — the agent can evolve this to class-specific heads)
            pred_boxes = [
                (int(top_k_labels[i][k]), px1, py1, px2, py2)
                for k in range(TOP_K)
            ]
            all_predictions.append({
                "image_id": image_id,
                "predictions": pred_boxes,
                "orig_shape": (int(orig_h), int(orig_w))
            })

        processed += batch_size_actual
        if processed % 5000 == 0:
            log.info(f"[Inference] Processed {processed}/{len(image_ids)} images")

    # --- 5. Write raw predictions JSON ---
    with open(predictions_json, "w") as f:
        json.dump(all_predictions, f)
    log.info(f"[Inference] Raw predictions saved to {predictions_json}")

    # --- 6. Write Kaggle submission CSV ---
    from imagenet.submission_generator import SubmissionGenerator
    gen = SubmissionGenerator()
    submission_path = gen.write_csv(all_predictions, submission_csv)
    log.info(f"[Inference] Submission CSV: {submission_path}")

    result = {
        "status": "success",
        "submission_csv_path": submission_path,
        "raw_predictions_path": predictions_json,
        "num_images_processed": len(all_predictions),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"[Inference] Done. Results written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--args_json", required=True)
    parser.add_argument("--output_path", required=True)
    cli = parser.parse_args()
    args = json.loads(cli.args_json)
    run_inference(args, cli.output_path)
