import argparse
import json
import os
import sys
import traceback

import librosa
import numpy as np
import pandas as pd
import pywt
from PIL import Image, ImageEnhance
from pyts.image import GramianAngularField
from skimage.transform import resize
from scipy.ndimage import gaussian_filter1d # For Dynamic GAF

# 1. Trova il percorso della cartella principale del progetto (due livelli sopra lo script corrente)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# 2. Aggiungi la cartella principale al percorso di ricerca di Python
sys.path.append(PROJECT_ROOT)

# 3. Ora l'importazione funzionerà correttamente
from state_manager import PERSISTENT_PATHS
from ui_logger import log

# 1. Importa il dizionario dei percorsi con un "fallback" di sicurezza
try:
    from state_manager import PERSISTENT_PATHS
except ImportError:
    # Se lo script viene eseguito da solo, calcola i percorsi da solo
    print("[Worker Fallback] 'state_manager' not found. Calculating paths relatively.")
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
    PERSISTENT_PATHS = {
        "raw_data_dir": os.path.join(_PROJECT_ROOT, "processed_data", "raw_data"),
        "generated_datasets_dir": os.path.join(_PROJECT_ROOT, "processed_data", "generated_datasets"),
    }


# 2. Rimuovi la variabile globale 'source_path' che era qui


def _load_from_csv(metadata: dict, source_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Carica i dati seguendo la configurazione originale basata su CSV."""
    all_spectra, all_labels = [], []
    print("--- Loading raw data from CSV files based on metadata ---")
    for config in metadata["file_configurations"]:
        for filename in config["paths"]:
            # Il 'source_path' corretto viene passato come argomento dalla funzione chiamante
            path = os.path.join(source_path, filename)
            if not os.path.exists(path) and os.path.exists(path + ".csv"):
                path += ".csv"
            try:
                df = pd.read_csv(path)
                spectra = df.iloc[:, 1:].fillna(0).values.astype(np.float32).T
                labels = np.array([config["category"]] * spectra.shape[0])
                all_spectra.append(spectra)
                all_labels.append(labels)
            except FileNotFoundError:
                print(f"  [WARNING] File not found and skipped: {path}")
                continue
    if not all_spectra:
        raise RuntimeError("No raw data loaded from CSV source.")
    return np.vstack(all_spectra), np.concatenate(all_labels)


def _load_from_tabular_csv(metadata: dict, source_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Carica dati tabulari (Kaggle style) dove ogni riga è un campione.
    Supporta la rimozione automatica della colonna 'id' e l'estrazione della label.
    """
    filename = metadata.get("train_file", "train.csv")
    path = os.path.join(source_path, filename)
    
    # Fallback to test file if train doesn't exist (e.g. in test_data_folder)
    if not os.path.exists(path):
        filename = metadata.get("test_file", "test.csv")
        path = os.path.join(source_path, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Neither train nor test CSV found in {source_path}")

    print(f"--- Loading tabular data from {filename} ---")
    df = pd.read_csv(path)
    
    # Dropping 'id' if exists
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    # Label identification
    label_col = metadata.get("label_column", "Heart Disease")
    
    if label_col in df.columns:
        X = df.drop(columns=[label_col]).fillna(0).values.astype(np.float32)
        y = df[label_col].values
    else:
        # Case for test set without labels
        print(f"  [INFO] Label column '{label_col}' not found. Returning dummy labels.")
        X = df.fillna(0).values.astype(np.float32)
        y = np.array(["Unknown"] * len(X))
        
    print(f"  Loaded {len(X)} samples with {X.shape[1]} features.")
    return X, y


def _load_from_numpy(metadata: dict, source_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Carica e unisce i dati da una o più coppie di file .npy specificate nei metadati."""
    print("--- Loading and merging raw data from .npy files based on metadata ---")
    all_data, all_labels = [], []
    if "file_pairs" not in metadata or not metadata["file_pairs"]:
        raise ValueError("Metadata for numpy_files must contain a 'file_pairs' list.")
    for pair in metadata["file_pairs"]:
        data_fn, labels_fn = pair.get("data_file"), pair.get("labels_file")
        if not data_fn or not labels_fn:
            print(f"  [WARNING] Skipping invalid file pair in metadata: {pair}")
            continue
        data_path = os.path.join(source_path, data_fn)
        labels_path = os.path.join(source_path, labels_fn)
        if not os.path.exists(data_path) or not os.path.exists(labels_path):
            raise FileNotFoundError(f"File pair not found: {data_fn}, {labels_fn}")
        print(f"  -> Loading {data_fn} and {labels_fn}...")
        all_data.append(np.load(data_path))
        all_labels.append(np.load(labels_path, allow_pickle=True))
    if not all_data:
        raise RuntimeError("No data could be loaded from the specified .npy files.")
    final_data = np.concatenate(all_data, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)
    print(
        f"Successfully merged {len(all_data)} file pairs. Final shapes: Data {final_data.shape}, Labels {final_labels.shape}"
    )
    return final_data, final_labels


def _load_from_image_folders(metadata: dict, source_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Carica immagini recursively da sottocartelle.
    Supporta dataset strutturati come 'train/class_a', 'test/class_a' ecc.
    Usa il nome della cartella genitore immediata come etichetta.
    Restituisce dati in formato uint8 [0, 255] per efficienza e compatibilità con Rescaling layer.
    """
    print(f"--- Loading raw data from image folders (Recursive) in: {source_path} ---")
    all_images, all_labels = [], []
    target_size = tuple(metadata.get("image_size", [64, 64]))
    use_grayscale = metadata.get("channels", 3) == 1
    
    # Supported extensions
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

    # Reserved subfolders hold OTHER splits (held-out / finetune / shifted). Loading
    # the current set must never descend into them, or images would leak across splits.
    reserved_split_dirs = {"test_data_folder", "finetune_data_folder", "shifted_test_folder"}

    for root, dirs, files in os.walk(source_path):
        # Prune reserved split subfolders from traversal (in-place edit of `dirs`).
        dirs[:] = [d for d in dirs if d not in reserved_split_dirs]
        # Determine relative path from source_path
        rel_path = os.path.relpath(root, source_path)
        
        # Skip top level root if it has images (unlabeled)
        if rel_path == ".":
            continue
            
        # FIX: Use the immediate folder name as the label
        # This allows structures like 'train/airplane' -> label 'airplane'
        # Previous logic using split(os.sep)[0] would result in 'train' -> skipped
        label = os.path.basename(root)
        
        # Filter out common non-class folders (only if they are the immediate folder)
        if label.lower() in ["train", "test", "val", "validation", "raw_data", "split_data", ".ipynb_checkpoints"]:
            print(f"  [DEBUG] Skipping non-class folder: {rel_path}")
            continue

        print(f"  [DEBUG] Found class folder: {label} (Path: {rel_path})")

        for filename in files:
            if filename.lower().endswith(valid_exts):
                try:
                    img_path = os.path.join(root, filename)
                    img = Image.open(img_path)
                    
                    # --- Image Enhancements ---
                    contrast = metadata.get("contrast", 1.0)
                    if contrast != 1.0:
                        img = ImageEnhance.Contrast(img).enhance(contrast)
                        
                    brightness = metadata.get("brightness", 1.0)
                    if brightness != 1.0:
                        img = ImageEnhance.Brightness(img).enhance(brightness)
                        
                    sharpness = metadata.get("sharpness", 1.0)
                    if sharpness != 1.0:
                        img = ImageEnhance.Sharpness(img).enhance(sharpness)
                        
                    color = metadata.get("color", 1.0)
                    if color != 1.0:
                        img = ImageEnhance.Color(img).enhance(color)
                    # -------------------------

                    img = img.resize(target_size)
                    if use_grayscale:
                        img = img.convert("L")
                    elif img.mode != "RGB":
                        img = img.convert("RGB")
                    
                    # Keep as uint8 [0, 255]
                    img_array = np.array(img, dtype=np.uint8)
                    
                    # Ensure shape is consistent for concatenation
                    expected_shape = (target_size[1], target_size[0], 1 if use_grayscale else 3)
                    if img_array.shape != expected_shape:
                        if use_grayscale and len(img_array.shape) == 2:
                             img_array = np.expand_dims(img_array, axis=-1)
                        else:
                             # Skip instead of crash
                             continue

                    all_images.append(img_array)
                    all_labels.append(label)
                except Exception as e:
                    print(f"  [WARNING] Could not load or process image {filename}: {e}")

    if not all_images:
        raise RuntimeError(f"No images loaded from source: {source_path}. Check extensions and permissions.")
        
    images_np = np.array(all_images) # Should be uint8
    
    # Expand dims for grayscale if needed (H, W) -> (H, W, 1)
    if len(images_np.shape) == 3 and use_grayscale:
        images_np = np.expand_dims(images_np, axis=-1)
        
    # Standardize RGB if needed (H, W, 3) - PIL usually handles this but safety check
    if len(images_np.shape) == 3 and not use_grayscale and images_np.shape[-1] != 3:
         pass

    print(f"Loaded {len(images_np)} images. Shape: {images_np.shape}. Labels: {len(all_labels)}")
    
    # --- Class Distribution Summary ---
    unique_labels, counts = np.unique(np.array(all_labels), return_counts=True)
    print("--- Class Distribution ---")
    for label, count in zip(unique_labels, counts):
        print(f"  Class '{label}': {count} samples")
    print("--------------------------")
    
    return images_np, np.array(all_labels)


def _load_from_image_csv(metadata: dict, source_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load images that live FLAT in one or more directories, with labels supplied
    by a CSV (id -> class). Lets a dataset be used AS-IS — e.g. the standard Kaggle
    HAM10000 dump (HAM10000_images_part_1/2 + HAM10000_metadata.csv) — without
    restructuring it into per-class folders.

    Required metadata keys:
        label_csv     : CSV file (relative to source_path) with id and label columns
        id_column     : column holding the image id (e.g. "image_id")
        label_column  : column holding the class label (e.g. "dx")
        image_dirs    : list of subfolders (relative to source_path) holding the images
    Optional: image_exts (default [.jpg,.jpeg,.png]), image_size, channels,
              contrast/brightness/sharpness/color (PIL enhancements).
    Returns uint8 images [0,255] (consumed by the 2D_IMAGE builder's Rescaling layer).
    """
    import pandas as pd

    label_csv = metadata.get("label_csv")
    id_col = metadata.get("id_column", "image_id")
    label_col = metadata.get("label_column", "label")
    image_dirs = metadata.get("image_dirs", ["."])
    exts = tuple(e.lower() for e in metadata.get("image_exts", [".jpg", ".jpeg", ".png"]))
    target_size = tuple(metadata.get("image_size", [64, 64]))
    use_grayscale = metadata.get("channels", 3) == 1

    if not label_csv:
        raise ValueError("image_csv metadata requires a 'label_csv' field.")
    csv_path = os.path.join(source_path, label_csv)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"label_csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    for col in (id_col, label_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {label_csv}. Available: {list(df.columns)}")

    # Build id -> filepath index by scanning each declared image dir RECURSIVELY.
    # Recursive scan makes this robust to however the archive was unpacked
    # (2 folders, 4 folders, nested) — labels come from the CSV by image_id, so
    # indexing extra files is harmless (only ids present in the CSV are loaded).
    index: dict[str, str] = {}
    for d in image_dirs:
        ddir = os.path.join(source_path, d)
        if not os.path.isdir(ddir):
            print(f"  [image_csv WARN] image_dir not found, skipping: {ddir}")
            continue
        for r, _subdirs, files in os.walk(ddir):
            for fn in files:
                stem, ext = os.path.splitext(fn)
                if ext.lower() in exts:
                    full_path = os.path.join(r, fn)
                    # Index by BOTH the stem and the full filename, so the CSV id
                    # column may be either "ISIC_0027419" (HAM10000) or
                    # "PAT_1516_1765_530.png" (PAD-UFES-20) and still resolve.
                    index.setdefault(stem, full_path)
                    index.setdefault(fn, full_path)
    if not index:
        raise RuntimeError(f"No images {exts} found in {image_dirs} under {source_path}.")

    contrast   = metadata.get("contrast", 1.0)
    brightness = metadata.get("brightness", 1.0)
    sharpness  = metadata.get("sharpness", 1.0)
    color      = metadata.get("color", 1.0)

    all_images, all_labels, missing = [], [], 0
    for _idx, row in df.iterrows():
        img_id = str(row[id_col])
        path = index.get(img_id)
        if path is None:
            missing += 1
            continue
        try:
            img = Image.open(path)
            if contrast != 1.0:
                img = ImageEnhance.Contrast(img).enhance(contrast)
            if brightness != 1.0:
                img = ImageEnhance.Brightness(img).enhance(brightness)
            if sharpness != 1.0:
                img = ImageEnhance.Sharpness(img).enhance(sharpness)
            if color != 1.0:
                img = ImageEnhance.Color(img).enhance(color)
            img = img.resize(target_size)
            img = img.convert("L") if use_grayscale else (img if img.mode == "RGB" else img.convert("RGB"))
            arr = np.array(img, dtype=np.uint8)
            if use_grayscale and arr.ndim == 2:
                arr = arr[..., np.newaxis]
            all_images.append(arr)
            all_labels.append(str(row[label_col]))
        except Exception as e:
            print(f"  [image_csv WARN] could not load {path}: {e}")

    if not all_images:
        raise RuntimeError("image_csv: no images loaded — check id/label columns and image_dirs.")
    if missing:
        print(f"  [image_csv] {missing} CSV rows had no matching image file (skipped).")

    images_np = np.array(all_images, dtype=np.uint8)
    labels_np = np.array(all_labels)
    print(f"--- image_csv: loaded {len(images_np)} images, shape {images_np.shape} ---")
    unique_labels, counts = np.unique(labels_np, return_counts=True)
    print("--- Class Distribution ---")
    for lbl, cnt in zip(unique_labels, counts):
        print(f"  Class '{lbl}': {cnt} samples")
    print("--------------------------")
    return images_np, labels_np


def load_and_preprocess_raw_data(source_name: str) -> tuple[tuple[np.ndarray, np.ndarray], str]:
    """Funzione router che carica i dati grezzi in base al tipo specificato nel metadata.json."""
    if os.path.isabs(source_name) and os.path.exists(source_name):
         source_path = source_name
    else:
         source_path = os.path.join(PERSISTENT_PATHS["raw_data_dir"], source_name)
    
    metadata_path = os.path.join(source_path, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json not found in data source: {source_name}")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    data_type = metadata.get("type")
    if data_type == "csv":
        return _load_from_csv(metadata, source_path), "timeseries"
    elif data_type == "tabular_csv":
        return _load_from_tabular_csv(metadata, source_path), "timeseries"
    elif data_type in ("numpy_files", "numpy"):
        data, labels = _load_from_numpy(metadata, source_path)
        data_nature = "image" if len(data.shape) >= 3 else "timeseries"
        return (data, labels), data_nature
    elif data_type == "image_folders":
        return _load_from_image_folders(metadata, source_path), "image"
    elif data_type == "image_csv":
        return _load_from_image_csv(metadata, source_path), "image"
    else:
        raise ValueError(f"Unsupported data type '{data_type}' in metadata.")


# --- FUNZIONI DI RAPPRESENTAZIONE (invariate) ---
def represent_as_1d_array(spectra):
    return spectra.reshape(spectra.shape[0], spectra.shape[1], 1)


def represent_as_spectrogram(spectra, n_fft, hop_length):
    spectrograms = [
        librosa.amplitude_to_db(np.abs(librosa.stft(s.flatten(), n_fft=n_fft, hop_length=hop_length)), ref=np.max)
        for s in spectra
    ]
    max_width = max(s.shape[1] for s in spectrograms)
    padded = [np.pad(s, ((0, 0), (0, max_width - s.shape[1])), "constant") for s in spectrograms]
    return np.array(padded)[..., np.newaxis]


def represent_as_2d_gaf(spectra, image_size):
    gaf = GramianAngularField(image_size=image_size, method="summation")
    spectra_scaled = np.array(
        [
            2 * (s - np.min(s)) / (np.max(s) - np.min(s) + 1e-9) - 1 if np.max(s) != np.min(s) else np.zeros_like(s)
            for s in spectra
        ]
    )
    gaf_images = gaf.fit_transform(spectra_scaled)
    return gaf_images[..., np.newaxis]


def represent_as_cwt_scalogram(spectra, image_size=64):
    scales = np.arange(1, image_size + 1)
    all_scalograms = [
        resize(np.abs(pywt.cwt(s.flatten(), scales, "morl")[0]), (image_size, image_size), anti_aliasing=True)
        for s in spectra
    ]
    return np.array(all_scalograms)[..., np.newaxis]


def represent_as_3d_dynamic_cwt(spectra, n_frames=8, image_size=64):
    """
    3D_DYNAMIC_CWT (Option A — multi-resolution zoom).

    Each frame covers a different scale band of the CWT, sweeping from
    coarse (low-frequency, global structure) to fine (high-frequency, peak detail).
    Output shape: (N, n_frames, image_size, image_size, 1)

    The full scale range [1 .. image_size*4] is divided into n_frames equal
    logarithmic bands. Frame 0 = coarsest, frame n_frames-1 = finest.
    """
    n_total_scales = image_size * 4
    log_edges = np.logspace(0, np.log10(n_total_scales), n_frames + 1)

    videos = []
    for s in spectra:
        sig = s.flatten()
        frames = []
        for i in range(n_frames):
            lo = max(1, int(log_edges[i]))
            hi = max(lo + 1, int(log_edges[i + 1]))
            band_scales = np.arange(lo, hi)
            coeffs, _ = pywt.cwt(sig, band_scales, "morl")
            frame = resize(np.abs(coeffs), (image_size, image_size), anti_aliasing=True)
            frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-9)
            frames.append(frame)
        videos.append(np.stack(frames, axis=0))
    return np.array(videos)[..., np.newaxis]


def represent_as_3d_wavelet_cwt(spectra, n_frames=8, image_size=64):
    """
    3D_WAVELET_CWT (Option B — multi-wavelet ensemble).

    Each frame is the CWT scalogram computed with a different mother wavelet,
    capturing different spectral shapes. The wavelet sequence is fixed so that
    frame index is reproducible across samples.
    Output shape: (N, n_frames, image_size, image_size, 1)

    Wavelet pool (cycled if n_frames > len(pool)):
      morl, mexh, gaus2, gaus4, cgau1, cgau3, shan, fbsp
    """
    _WAVELET_POOL = ["morl", "mexh", "gaus2", "gaus4", "cgau1", "cgau3", "shan", "fbsp"]
    scales = np.arange(1, image_size + 1)
    wavelets = [_WAVELET_POOL[i % len(_WAVELET_POOL)] for i in range(n_frames)]

    videos = []
    for s in spectra:
        sig = s.flatten()
        frames = []
        for w in wavelets:
            try:
                coeffs, _ = pywt.cwt(sig, scales, w)
            except Exception:
                # Some complex wavelets return complex coeffs — take abs
                coeffs, _ = pywt.cwt(sig, scales, "morl")
            frame = resize(np.abs(coeffs), (image_size, image_size), anti_aliasing=True)
            frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-9)
            frames.append(frame)
        videos.append(np.stack(frames, axis=0))
    return np.array(videos)[..., np.newaxis]


def represent_as_3d_video(spectra, num_segments, n_fft=32, hop_length=8):
    videos = []
    for spectrum in spectra:
        segment_length = len(spectrum) // num_segments
        if segment_length < n_fft:
            raise ValueError(f"Segment length {segment_length} < n_fft {n_fft}. Sequence too short for 3D Video.")
        frames = [
            librosa.amplitude_to_db(
                np.abs(
                    librosa.stft(
                        spectrum[i * segment_length : (i + 1) * segment_length].flatten(),
                        n_fft=n_fft,
                        hop_length=hop_length,
                    )
                ),
                ref=np.max,
            )
            for i in range(num_segments)
        ]
        max_width = max(f.shape[1] for f in frames)
        padded = [np.pad(f, ((0, 0), (0, max_width - f.shape[1])), "constant") for f in frames]
        videos.append(np.stack(padded, axis=0))
    if not videos:
        raise RuntimeError("No 3D spectrogram videos created. Segments might be too short.")
    return np.array(videos)[..., np.newaxis]


def represent_as_3d_gaf_video(spectra, num_segments, image_size):
    gaf = GramianAngularField(image_size=image_size, method="summation")
    videos = []
    for spectrum in spectra:
        segment_length = len(spectrum) // num_segments
        if segment_length < 2:
            raise ValueError(f"Segment length {segment_length} too short for 3D GAF Video.")
        frames = []
        for i in range(num_segments):
            segment = spectrum[i * segment_length : (i + 1) * segment_length]
            seg_scaled = (
                2 * (segment - np.min(segment)) / (np.max(segment) - np.min(segment) + 1e-9) - 1
                if np.max(segment) != np.min(segment)
                else np.zeros_like(segment)
            )
            frames.append(gaf.fit_transform(seg_scaled.reshape(1, -1))[0])
        videos.append(np.stack(frames, axis=0))
    if not videos:
        raise RuntimeError("No 3D GAF videos created. Segments might be too short.")
    return np.array(videos)[..., np.newaxis]


def represent_as_3d_dynamic_gaf(spectra, n_frames=60, image_size=64, method='summation'):
    """
    Generates a 3D Video from 1D spectra using Gaussian Smoothing Evolution.
    Method:
    - 'summation': 100% GASF with Sigma 5.0 -> 0.0
    - 'difference': 100% GADF with Sigma 5.0 -> 0.0
    - 'hybrid': 50% GASF / 50% GADF (Sigma 5.0->0.0 for each half)
    """
    n_frames_original = n_frames
    n_frames = max(n_frames, 2)
    if n_frames != n_frames_original:
        print(f"[Warning] 3D_DYNAMIC_GAF requires at least 2 frames. Clamped n_frames from {n_frames_original} to {n_frames}.")

    print(f"[Dynamic GAF] Generating {n_frames} frames (Size: {image_size}x{image_size}) using method='{method}'...")
    videos = []
    
    # Helper to generate one video for one spectrum
    def _generate_video_single(signal_1d):
        frames = []
        signal_1d = signal_1d.reshape(1, -1) # Format (1, Points)
        
        # Sigma range
        start_sigma = 5.0
        end_sigma = 0.0
        
        # Logic for Hybrid vs Single
        loop_frames = n_frames
        modes = [method]
        
        if method == 'hybrid':
            loop_frames = n_frames // 2
            modes = ['summation', 'difference']
            
        for m in modes:
            for i in range(loop_frames):
                # Calculate Sigma
                progress = i / max(loop_frames - 1, 1)
                sigma = start_sigma * (1 - progress)
                
                # Apply Smoothing
                if sigma > 0.01:
                    sig_smooth = gaussian_filter1d(signal_1d, sigma=sigma, axis=1)
                else:
                    sig_smooth = signal_1d
                
                # SERS signals are usually 1000+ points. image_size is 64. So Pyts GAF handles downscaling via PAA internally.
                transformer = GramianAngularField(image_size=image_size, method=m)
                try:
                    img = transformer.fit_transform(sig_smooth)[0]
                except Exception as e:
                    # Fallback padding if something goes wrong
                    print(f"GAF Error: {e}. Padding signal.")
                    padded = np.pad(sig_smooth, ((0,0), (0, image_size)), 'edge')
                    img = transformer.fit_transform(padded)[0]
                    
                frames.append(img)
                
        return np.stack(frames, axis=0) # (Frames, H, W)

    count = 0
    for s in spectra:
        # Removed outer try-except to prevent silent dropping of samples
        vid = _generate_video_single(s)
        videos.append(vid)
        count += 1
        if count % 100 == 0:
            print(f"  Processed {count}/{len(spectra)}...", end='\r')
            
    print(f"  Done. Generated {len(videos)} videos.")
    if not videos:
        raise RuntimeError("No 3D Dynamic GAF videos created.")
        
    return np.array(videos)[..., np.newaxis] # (Samples, Frames, H, W, 1)


# --- FINETUNE HELPER ---
def _try_generate_finetune(source_path: str, dataset_path: str, args, params) -> bool:
    """
    Look for X_finetune.npy + y_finetune.npy directly (no metadata.json required).
    Searches the source folder and well-known sibling folders. If found, applies
    the representation transform and saves with _X_finetune.npy suffix.
    Returns True if a finetune split was generated, False otherwise.
    """
    candidate_dirs = [
        source_path,
        os.path.join(source_path, "finetune_data_folder"),
        os.path.join(source_path, "test_data_folder_clinical"),
    ]
    x_path = y_path = None
    for d in candidate_dirs:
        x_cand = os.path.join(d, "X_finetune.npy")
        y_cand = os.path.join(d, "y_finetune.npy")
        if os.path.exists(x_cand) and os.path.exists(y_cand):
            x_path, y_path = x_cand, y_cand
            print(f"--- FINETUNE SET DISCOVERED: {x_path} ---")
            break
    if x_path is None:
        return False

    try:
        raw_data = np.load(x_path)
        labels = np.load(y_path, allow_pickle=True)
        if len(raw_data) != len(labels):
            raise RuntimeError(f"[FINETUNE_SET] X/y length mismatch: {len(raw_data)} vs {len(labels)}")

        # Save labels
        labels_out = os.path.join(dataset_path, "labels_finetune.npy")
        np.save(labels_out, labels)
        print(f"[FINETUNE_SET] Labels saved -> labels_finetune.npy")

        # Apply representation transform
        data_to_save = _apply_representation(raw_data, args.representation_type, params)

        rep_out = os.path.join(dataset_path, f"{args.representation_type}_X_finetune.npy")
        np.save(rep_out, data_to_save)
        print(f"[FINETUNE_SET] Representation saved -> {os.path.basename(rep_out)}  shape={data_to_save.shape}")
        return True
    except Exception as e:
        print(f"[FINETUNE_SET] Skipped due to error: {e}")
        return False


# Representations that are genuine 2-D images (no 1-D transform is applied).
IMAGE_NATIVE_REPS = ("2D_IMAGE", "2D_GENERIC_IMAGE")
# Representations derived from a 1-D signal (GAF / CWT / spectrogram / 1D / 3D-dynamic).
# They are mathematically undefined for image input and MUST fail loudly here —
# never silently pass the raw image through under a transform's name (that would log
# e.g. "2D_GAF=X%" for a model that never actually saw a GAF), which corrupts results.
SIGNAL_DERIVED_REPS = (
    "1D_CNN", "2D_SPECTROGRAM", "2D_GAF", "2D_CWT_SCALOGRAM",
    "3D_VIDEO", "3D_GAF_VIDEO", "3D_DYNAMIC_GAF", "3D_DYNAMIC_CWT", "3D_WAVELET_CWT",
)


def _assert_rep_matches_modality(rep_type: str, is_image_like: bool, shape) -> None:
    """Reject a modality/representation mismatch with an actionable error instead of
    a silent passthrough or an opaque downstream crash. Lets the self-healing agent
    learn that a 1-D transform is not admissible for an image dataset."""
    if is_image_like and rep_type in SIGNAL_DERIVED_REPS:
        raise ValueError(
            f"representation_type '{rep_type}' is defined only for 1-D signals "
            f"(spectra / time-series), but the input is image-like with shape {shape}. "
            f"For image datasets use representation_type='2D_IMAGE'. "
            f"(GAF / CWT / spectrogram / 3D-dynamic transforms have no single time axis on an image.)"
        )


def _apply_representation(raw_data: np.ndarray, rep_type: str, params: dict) -> np.ndarray:
    """Pure transform: raw spectra/images -> representation array. No I/O."""
    is_image_like = len(raw_data.shape) >= 3
    # Genuine image representation: pass the image through untouched.
    if is_image_like and rep_type in IMAGE_NATIVE_REPS:
        return raw_data
    # Modality guard: a 1-D transform on image input is invalid — fail loudly.
    _assert_rep_matches_modality(rep_type, is_image_like, raw_data.shape)
    if rep_type == "1D_CNN":
        return represent_as_1d_array(raw_data)
    if rep_type == "2D_SPECTROGRAM":
        return represent_as_spectrogram(raw_data, params["n_fft"], params["hop_length"])
    if rep_type == "2D_GAF":
        return represent_as_2d_gaf(raw_data, params["gaf_image_size"])
    if rep_type == "2D_CWT_SCALOGRAM":
        return represent_as_cwt_scalogram(raw_data, params.get("gaf_image_size", 64))
    if rep_type == "3D_VIDEO":
        return represent_as_3d_video(raw_data, params["video_num_segments"])
    if rep_type == "3D_GAF_VIDEO":
        return represent_as_3d_gaf_video(raw_data, params["video_num_segments"], params["video_gaf_image_size"])
    if rep_type == "3D_DYNAMIC_GAF":
        return represent_as_3d_dynamic_gaf(
            raw_data,
            params.get("video_num_segments", 60),
            params.get("video_gaf_image_size", 64),
            params.get("gaf_method", "summation"),
        )
    if rep_type == "3D_DYNAMIC_CWT":
        return represent_as_3d_dynamic_cwt(
            raw_data, params.get("video_num_segments", 8), params.get("video_gaf_image_size", 64))
    if rep_type == "3D_WAVELET_CWT":
        return represent_as_3d_wavelet_cwt(
            raw_data, params.get("video_num_segments", 8), params.get("video_gaf_image_size", 64))
    raise ValueError(f"Unknown representation_type '{rep_type}' in _apply_representation.")


# --- HELPER FUNZIONE ESTRAZIONE ---
def _process_single_folder(source_folder: str, dataset_path: str, args, params, is_test_set=False, kind=None):
    """
    Processes a single raw data folder and saves the representation into the target dataset_path.

    Args:
        kind: optional explicit label — "train" (default), "test", or "finetune".
              When provided overrides legacy is_test_set. Controls file naming.
    """
    # Resolve kind from legacy is_test_set if not provided
    if kind is None:
        kind = "test" if is_test_set else "train"

    if kind == "train":
        prefix = "TRAIN_SET"
        labels_file_name = "labels.npy"
        representation_file_name = f"{args.representation_type}_data.npy"
        overwrite_warn = True
    elif kind == "test":
        prefix = "TEST_SET"
        labels_file_name = "labels_test_heldout.npy"
        representation_file_name = f"{args.representation_type}_X_test_heldout.npy"
        overwrite_warn = False
    elif kind == "finetune":
        prefix = "FINETUNE_SET"
        labels_file_name = "labels_finetune.npy"
        representation_file_name = f"{args.representation_type}_X_finetune.npy"
        overwrite_warn = False
    elif kind == "shifted_test":
        prefix = "SHIFTED_TEST_SET"
        labels_file_name = "labels_shifted_test.npy"
        representation_file_name = f"{args.representation_type}_X_shifted_test.npy"
        overwrite_warn = False
    else:
        raise ValueError(f"Unknown kind '{kind}'. Must be 'train', 'test', 'finetune', or 'shifted_test'.")

    print(f"[{prefix}] Processing source folder: {source_folder}")

    (raw_data, labels), data_nature = load_and_preprocess_raw_data(source_folder)
    log(f"[{prefix} DEBUG] Raw data loaded successfully. Shape: {raw_data.shape}, Nature: {data_nature}")

    if len(raw_data) != len(labels):
        raise RuntimeError(f"[{prefix}] Data length mismatch! Data={len(raw_data)}, Labels={len(labels)}")

    labels_path = os.path.join(dataset_path, labels_file_name)
    if os.path.exists(labels_path) and overwrite_warn:
        print(f"[{prefix}] Overwriting existing {labels_file_name} to ensure consistency.")
    np.save(labels_path, labels)
    print(f"[{prefix}] Labels saved to {labels_file_name}.")

    data_to_save = None

    if data_nature == "image":
        # Only genuine image representations are valid for image input. A 1-D
        # transform (GAF/CWT/spectrogram/3D-dynamic) must fail loudly here — never
        # silently pass the raw image through under the transform's name.
        if args.representation_type in IMAGE_NATIVE_REPS:
            print(f"[{prefix}] Raw data is '{data_nature}'. Using image directly for '{args.representation_type}'.")
            data_to_save = raw_data
        else:
            raise ValueError(
                f"[{prefix}] representation_type '{args.representation_type}' requires 1-D signal input, "
                f"but dataset '{getattr(args, 'manifest_name', '?')}' is image-like (shape {raw_data.shape}). "
                f"Use representation_type='2D_IMAGE' for image datasets."
            )
    elif data_nature == "timeseries":
        print(f"[{prefix}] Raw data is '{data_nature}'. Applying transformations...")
        if args.representation_type == "1D_CNN":
            data_to_save = represent_as_1d_array(raw_data)
        elif args.representation_type == "2D_SPECTROGRAM":
            data_to_save = represent_as_spectrogram(raw_data, params["n_fft"], params["hop_length"])
        elif args.representation_type == "2D_GAF":
            data_to_save = represent_as_2d_gaf(raw_data, params["gaf_image_size"])
        elif args.representation_type == "2D_CWT_SCALOGRAM":
            data_to_save = represent_as_cwt_scalogram(raw_data, params.get("gaf_image_size", 64))
        elif args.representation_type == "3D_VIDEO":
            data_to_save = represent_as_3d_video(raw_data, params["video_num_segments"])
        elif args.representation_type == "3D_GAF_VIDEO":
            data_to_save = represent_as_3d_gaf_video(raw_data, params["video_num_segments"], params["video_gaf_image_size"])
        elif args.representation_type == "3D_DYNAMIC_GAF":
            n_frames = params.get("video_num_segments", 60)
            img_size = params.get("video_gaf_image_size", 64)
            method = params.get("gaf_method", "summation")
            data_to_save = represent_as_3d_dynamic_gaf(raw_data, n_frames, img_size, method)
        elif args.representation_type == "3D_DYNAMIC_CWT":
            n_frames = params.get("video_num_segments", 8)
            img_size = params.get("video_gaf_image_size", 64)
            data_to_save = represent_as_3d_dynamic_cwt(raw_data, n_frames, img_size)
        elif args.representation_type == "3D_WAVELET_CWT":
            n_frames = params.get("video_num_segments", 8)
            img_size = params.get("video_gaf_image_size", 64)
            data_to_save = represent_as_3d_wavelet_cwt(raw_data, n_frames, img_size)
        else:
            raise ValueError(f"Unknown representation_type '{args.representation_type}' for timeseries data.")
    else:
        raise ValueError(f"Cannot generate '{args.representation_type}' from raw data of nature '{data_nature}'.")

    output_file_path = os.path.join(dataset_path, representation_file_name)
    np.save(output_file_path, data_to_save)
    print(f"[{prefix}] Representation saved to {representation_file_name}.")
    
    return output_file_path

# --- WORKER PRINCIPALE (MODIFICATO) ---
def main(args):
    result = {}
    try:
        # 3. Usa il percorso corretto per i dataset generati
        dataset_path = os.path.join(PERSISTENT_PATHS["generated_datasets_dir"], args.manifest_name)
        os.makedirs(dataset_path, exist_ok=True)

        print(f"--- Running Modular Data Pipeline for Manifest: {args.manifest_name} ---")
        print(f"--- Using Raw Data Source: {args.raw_data_source} ---")
        print(f"--- Generating Representation: {args.representation_type} ---")

        params = json.loads(args.params_json)
        
        # Determine the absolute source path of the raw data
        if os.path.isabs(args.raw_data_source) and os.path.exists(args.raw_data_source):
             source_path = args.raw_data_source
        else:
             source_path = os.path.join(PERSISTENT_PATHS["raw_data_dir"], args.raw_data_source)
             
        # 1. Process Main Dataset
        output_file_path = _process_single_folder(source_path, dataset_path, args, params, is_test_set=False)
        message = f"Representation '{args.representation_type}' generated for '{args.manifest_name}'."
        
        # 2. Process Explicit Test Set (If exists)
        test_folder_path = os.path.join(source_path, "test_data_folder")
        if os.path.exists(test_folder_path):
            print(f"--- DYNAMIC TEST SET DISCOVERED: {test_folder_path} ---")
            _process_single_folder(test_folder_path, dataset_path, args, params, kind="test")
            message += f" (Explicit test set also generated)."

        # 3. Process Explicit Finetune Set (If exists)
        # Two discovery paths (in priority order):
        #   a) finetune_data_folder/ has a metadata.json -> _process_single_folder (supports CSV + NPY)
        #   b) legacy NPY discovery: X_finetune.npy + y_finetune.npy directly in known locations
        finetune_generated = False
        finetune_data_folder = os.path.join(source_path, "finetune_data_folder")
        finetune_meta = os.path.join(finetune_data_folder, "metadata.json")
        if os.path.exists(finetune_data_folder) and os.path.exists(finetune_meta):
            print(f"--- FINETUNE SET DISCOVERED (metadata): {finetune_data_folder} ---")
            try:
                _process_single_folder(finetune_data_folder, dataset_path, args, params, kind="finetune")
                finetune_generated = True
                message += f" (Finetune set also generated)."
            except Exception as ft_e:
                print(f"[FINETUNE_SET] metadata-based processing failed: {ft_e}. Trying NPY discovery...")
                finetune_generated = _try_generate_finetune(source_path, dataset_path, args, params)
                if finetune_generated:
                    message += f" (Finetune set also generated)."
        else:
            # Fallback: look for X_finetune.npy + y_finetune.npy directly
            finetune_generated = _try_generate_finetune(source_path, dataset_path, args, params)
            if finetune_generated:
                message += f" (Finetune set also generated)."

        # 4. Process Shifted Test Set (If exists) — different distribution
        # from training (e.g. aged, clinical, different acquisition session).
        # Reported as a separate diagnostic metric by training_worker so the
        # agent can see distribution-shift gap without it polluting the
        # primary HO optimization signal.
        shifted_generated = False
        shifted_folder = os.path.join(source_path, "shifted_test_folder")
        if os.path.exists(shifted_folder):
            print(f"--- SHIFTED TEST SET DISCOVERED: {shifted_folder} ---")
            try:
                _process_single_folder(shifted_folder, dataset_path, args, params, kind="shifted_test")
                message += f" (Shifted test set also generated)."
                shifted_generated = True
            except Exception as shift_e:
                print(f"[SHIFTED_TEST_SET] Skipped due to error: {shift_e}")

        result = {
            "status": "completed",
            "message": message,
            "data_file_path": output_file_path,
            "has_finetune_split": finetune_generated,
            "has_shifted_test": shifted_generated,
        }

    except Exception as e:
        error_trace = traceback.format_exc()

        log(f"[Data Worker CRASH] The data pipeline failed. Full traceback:\n{error_trace}")
        result = {"status": "error", "message": str(e), "traceback": traceback.format_exc()}

    with open(args.output_path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Worker for modular data processing pipeline.")
    parser.add_argument(
        "--raw_data_source", type=str, required=True, help="Name of the raw data source folder to use."
    )
    parser.add_argument("--manifest_name", type=str, required=True)
    parser.add_argument("--representation_type", type=str, required=True)
    parser.add_argument("--params_json", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    parsed_args = parser.parse_args()
    main(parsed_args)
