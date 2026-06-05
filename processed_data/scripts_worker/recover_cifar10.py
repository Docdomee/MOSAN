import numpy as np
import os
import pickle
from PIL import Image
import sys

# Define standard CIFAR-10 class names
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def unpickle(file):
    """Unpickles a CIFAR-10 binary file."""
    print(f"Loading batch: {file}")
    with open(file, 'rb') as fo:
        # CIFAR-10 python version uses bytes encoding
        d = pickle.load(fo, encoding='bytes')
    return d

def save_images_from_batch(batch_data, batch_labels, filenames, output_dir, prefix=""):
    """
    Converts raw batch data to images and saves them.
    batch_data: (10000, 3072) numpy array
    batch_labels: list of integers
    """
    # Reshape: (N, 3, 32, 32) -> Transpose to (N, 32, 32, 3) for RGB
    raw_float = np.array(batch_data, dtype=float) / 255.0
    images = batch_data.reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1))
    
    for i, (img_array, label_idx) in enumerate(zip(images, batch_labels)):
        class_name = CLASS_NAMES[label_idx]
        
        # Create class folder if it doesn't exist
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Construct filename
        # Use provided filename from batch if available, else generic
        # filenames in batch are bytes
        original_fname = filenames[i].decode('utf-8') if filenames else f"img_{i}.png"
        save_name = f"{prefix}_{original_fname}"
        
        # Save
        save_path = os.path.join(class_dir, save_name)
        
        # Convert to PIL Image
        img = Image.fromarray(img_array)
        img.save(save_path)
        
    print(f"  Saved {len(images)} images to {output_dir}")

def main():
    # Paths (Hardcoded based on user context or arguments)
    # Default to relative paths if run from processed_data/scripts_worker/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # processed_data
    raw_data_dir = os.path.join(base_dir, "raw_data")
    
    source_dir = os.path.join(raw_data_dir, "cifar10_raw")
    output_dir = os.path.join(raw_data_dir, "cifar10_recovered_full")
    
    if not os.path.exists(source_dir):
        print(f"Error: Source directory not found: {source_dir}")
        return

    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Process Training Batches
    for i in range(1, 6):
        batch_file = os.path.join(source_dir, f"data_batch_{i}")
        if os.path.exists(batch_file):
            data_dict = unpickle(batch_file)
            # Keys: b'batch_label', b'labels', b'data', b'filenames'
            save_images_from_batch(
                data_dict[b'data'], 
                data_dict[b'labels'], 
                data_dict.get(b'filenames'),
                os.path.join(output_dir, "train"), # Merge all training batches into 'train' superfolder? 
                # Actually, standard pipeline usually expects just class folders.
                # But typically we want a train/test split.
                # CIFAR-10 has distinct test_batch. 
                # Let's create 'train' and 'test' folders at root.
                prefix=f"batch{i}"
            )
        else:
            print(f"Warning: Batch file missing: {batch_file}")

    # Process Test Batch
    test_file = os.path.join(source_dir, "test_batch")
    if os.path.exists(test_file):
        data_dict = unpickle(test_file)
        save_images_from_batch(
            data_dict[b'data'], 
            data_dict[b'labels'], 
            data_dict.get(b'filenames'),
            os.path.join(output_dir, "test"),
            prefix="test"
        )
    else:
        print(f"Warning: Test batch file missing: {test_file}")

    print("\nRecovery Complete!")
    print(f"Images are located in: {output_dir}")
    print("Structure:")
    print(f"  {output_dir}/train/<class_name>/...")
    print(f"  {output_dir}/test/<class_name>/...")

if __name__ == "__main__":
    main()
