import os
import sys
import nibabel as nib
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from multiprocessing import Pool
from tqdm import tqdm

# Make the repo root importable so that `paths_config` resolves regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths_config import (
    CTRATE_RAW_VALID_DIR,
    CTRATE_VALID_METADATA,
    CTRATE_VALID_IMAGES,
    PREPROCESS_NUM_WORKERS,
    require,
)

def read_nii_files(directory):
    """
    Retrieve paths of all NIfTI files in the given directory.

    Args:
    directory (str): Path to the directory containing NIfTI files.

    Returns:
    list: List of paths to NIfTI files.
    """
    nii_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.nii.gz'):
                nii_files.append(os.path.join(root, file))
    return nii_files

def read_nii_data(file_path):
    """
    Read NIfTI file data.

    Args:
    file_path (str): Path to the NIfTI file.

    Returns:
    np.ndarray: NIfTI file data.
    """
    try:
        nii_img = nib.load(file_path)
        nii_data = nii_img.get_fdata()
        return nii_data
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing.

    Args:
    array (torch.Tensor): Input array to be resized.
    current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
    target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

    Returns:
    np.ndarray: Resized array.
    """
    # Calculate new dimensions
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    # Resize the array
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array

def process_file(file_path):
    """
    Process a single NIfTI file.

    Args:
    file_path (str): Path to the NIfTI file.

    Returns:
    None
    """
    

    file_name = os.path.basename(file_path)

    try:
        row = df[df['VolumeName'] == file_name]
        slope = 1 # float(row["RescaleSlope"].iloc[0])
        intercept = float(row["RescaleIntercept"].iloc[0])
        xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
        z_spacing = float(row["ZSpacing"].iloc[0])
    except:
        print(file_name)
        return

    save_folder = CTRATE_VALID_IMAGES #save folder for preprocessed
    
    folder_path_new = os.path.join(save_folder, "valid_" + file_name.split("_")[1], "valid_" + file_name.split("_")[1] + file_name.split("_")[2]) #folder name for train or validation
    os.makedirs(folder_path_new, exist_ok=True)
    file_name = file_name.split(".")[0]+".npz"

    save_path = os.path.join(folder_path_new, file_name)

    if os.path.exists(save_path):
        return

    img_data = read_nii_data(file_path)
    if img_data is None:
        print(f"Read {file_path} unsuccessful. Passing")
        return
    
    os.makedirs(folder_path_new, exist_ok=True)
    


    # Define the target spacing values
    target_x_spacing = 0.75
    target_y_spacing = 0.75
    target_z_spacing = 1.5

    # target_x_spacing = 1.5
    # target_y_spacing = 1.5
    # target_z_spacing = 3

    current = (z_spacing, xy_spacing, xy_spacing)
    target = (target_z_spacing, target_x_spacing, target_y_spacing)

    img_data = slope * img_data + intercept
    hu_min, hu_max = -1000, 1000
    img_data = np.clip(img_data, hu_min, hu_max)
    img_data = ((img_data / 1000)).astype(np.float32)

    img_data = img_data.transpose(2, 0, 1)
    tensor = torch.tensor(img_data)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    resized_array = resize_array(tensor, current, target)
    resized_array = resized_array[0][0]

    # save_folder = "/jizhicfs/datalh/dataset/CTRATE/data_volumes/dataset/valid_preprocessed_15_3_r/" #save folder for preprocessed
    
    # save_path = os.path.join(folder_path_new, file_name)
    np.savez(save_path, resized_array)

def process_file_pillar(file_path, save_folder = CTRATE_VALID_IMAGES.rstrip("/") + "_pillar"): 
    """
    Process a single NIfTI file for Pillar-0/Atlas encoder.
    Key differences from original:
    - Keep raw HU (no clip/normalize)
    - Resample to isotropic 1.25mm spacing
    - Center crop/pad to 256x256x256
    - Save as .npy for dynamic 11-channel generation at inference
    """
    file_name = os.path.basename(file_path)

    try:
        row = df[df['VolumeName'] == file_name]
        slope = 1  # Keep as 1 unless your DICOM has non-unit slope
        intercept = float(row["RescaleIntercept"].iloc[0])
        xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
        z_spacing = float(row["ZSpacing"].iloc[0])
    except Exception as e:
        print(f"[Error] Meta {file_name} | {e}")
        with open("error.txt", "a") as f:
            f.write(f"{file_name}\n")
        return
    
    # Save path setup (same folder structure style as your original)
    folder_path_new = os.path.join(save_folder, "valid_" + file_name.split("_")[1])
    os.makedirs(folder_path_new, exist_ok=True)
    save_path = os.path.join(folder_path_new, file_name.replace(".nii.gz", ".npy"))  # .npy for single-channel raw HU

    if os.path.exists(save_path):
        return
    
    # 1. Read and convert to raw HU (NO clip, NO normalize)
    img_data = read_nii_data(file_path)
    if img_data is None:
        print(f"Read {file_path} unsuccessful. Passing")
        return
    img_hu = (slope * img_data + intercept).astype(np.float32)  # Keep full dynamic range [-1000, +3000+]

    # 2. Transpose to match resize_array input format (Z, Y, X)
    img_hu = img_hu.transpose(2, 0, 1)  # Same as your original code
    
    # 3. Convert to tensor and add batch/channel dims for resize_array
    tensor = torch.tensor(img_hu)
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, Z, Y, X]

    # 4. Resample to isotropic 1.25mm using YOUR resize_array function
    current = (z_spacing, xy_spacing, xy_spacing)
    target = (1.25, 1.25, 1.25)  # Pillar-0 standard isotropic spacing
    
    try:
        resized_tensor = resize_array(tensor, current, target)
        img_resampled = resized_tensor[0][0] # Remove batch/channel dims, back to (Z, Y, X)
    except Exception as e:
        print(f"[Error] Resize {file_name} | {e}")
        return
    print(img_resampled.shape, img_resampled.min(), img_resampled.max())
    # 5. Center crop/pad to fixed 256x256x256 matrix (DOES NOT change spacing)
    target_shape = (256, 256, 256)
    padded = np.full(target_shape, -1000, dtype=np.float32)  # Fill background with air HU

    z, y, x = img_resampled.shape
    z_crop = min(z, 256); y_crop = min(y, 256); x_crop = min(x, 256)

    # Destination start (where to place data inside the 256^3 canvas)
    z_dst = (256 - z_crop) // 2
    y_dst = (256 - y_crop) // 2
    x_dst = (256 - x_crop) // 2

    # Source start (where to crop from in the original volume) — CENTER crop
    z_src = (z - z_crop) // 2
    y_src = (y - y_crop) // 2
    x_src = (x - x_crop) // 2

    padded[z_dst:z_dst + z_crop,
           y_dst:y_dst + y_crop,
           x_dst:x_dst + x_crop] = img_resampled[z_src:z_src + z_crop,
                                                  y_src:y_src + y_crop,
                                                  x_src:x_src + x_crop]
    # print(img_resampled.shape, img_resampled.min(), img_resampled.max())
    # 6. Save raw HU (inference-time windowing applied later)
    np.save(save_path, padded)
    # print(f"[Saved] {save_path} | Shape: {padded.shape} | HU range: [{padded.min():.1f}, {padded.max():.1f}]")

# Example usage:
if __name__ == "__main__":
    require("CTRATE_RAW_VALID_DIR", "CTRATE_VALID_IMAGES")
    split_to_preprocess = CTRATE_RAW_VALID_DIR #select the validation or test split
    nii_files = read_nii_files(split_to_preprocess)

    print(len(nii_files))

    df = pd.read_csv(CTRATE_VALID_METADATA) #select the metadata

    num_workers = PREPROCESS_NUM_WORKERS  # Number of worker processes

    # Process files using multiprocessing with tqdm progress bar.
    # NOTE: use `process_file` (produces the *.npz under CTRATE_VALID_IMAGES that
    # the dataset actually consumes). `process_file_pillar` above is kept only as
    # an optional variant for the Pillar-0/Atlas encoder and is NOT used here.
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap(process_file, nii_files), total=len(nii_files)))
