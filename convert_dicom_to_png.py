import os
import glob
import cv2
import dicomsdl
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import config


# ==========================================
# Worker Function (Must be at top level)
# ==========================================
def process_single_file(file_path):
    """
    Reads a single DICOM and saves it as PNG.
    Returns (success_bool, message).
    """
    try:
        # Define target path inside the worker to avoid passing it around
        # (Re-calculating this is cheap)
        base_name = os.path.basename(file_path).replace(".dcm", ".png")
        target_path = os.path.join(config.PNG_DIR, base_name)

        # Optimization: Skip if already exists (resume capability)
        if os.path.exists(target_path):
            return True, None

        # 1. Load DICOM
        dcm = dicomsdl.open(file_path)
        img = dcm.pixelData().astype(np.float32)

        # 2. Safety Normalization (The "Anti-Dark" Fix)
        img_min = img.min()
        img_max = img.max()
        epsilon = 1e-6
        img_norm = (img - img_min) / (img_max - img_min + epsilon)

        # 3. Convert to uint8 (0-255)
        img_uint8 = (img_norm * 255).astype(np.uint8)

        # 4. Save
        cv2.imwrite(target_path, img_uint8)
        return True, None

    except Exception as e:
        return False, f"{os.path.basename(file_path)}: {str(e)}"


# ==========================================
# Main Execution
# ==========================================
def main():
    # 1. Setup
    source_dir = config.DICOM_DIR
    target_dir = config.PNG_DIR
    os.makedirs(target_dir, exist_ok=True)

    # 2. Collect Files
    dicom_files = glob.glob(os.path.join(source_dir, "*.dcm"))
    total_files = len(dicom_files)

    if not dicom_files:
        print("No DICOM files found.")
        return

    print(f"Starting Parallel Conversion on {os.cpu_count()} cores...")
    print(f"Target: {target_dir}")

    # 3. Run Parallel Pool
    # max_workers=None defaults to the number of processors on your machine
    with ProcessPoolExecutor() as executor:
        # Map returns an iterator that yields results as they finish
        results = list(
            tqdm(
                executor.map(process_single_file, dicom_files),
                total=total_files,
                desc="Converting",
            )
        )

    # 4. Statistics
    success_count = sum(1 for r in results if r[0])
    errors = [r[1] for r in results if not r[0]]

    print("\n" + "=" * 30)
    print(f"Done. Success: {success_count}/{total_files}")
    if errors:
        print(f"Errors: {len(errors)}")
        # Print first 5 errors to avoid spamming console
        for e in errors[:5]:
            print(f" - {e}")


if __name__ == "__main__":
    # This protection is REQUIRED for multiprocessing on Windows/macOS
    main()
