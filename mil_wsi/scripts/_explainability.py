import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import h5py
from openslide import OpenSlide


def visualize_attention(all_attn_weights, all_filenames, predictions, data_path, suffix, threshold=0.5, patch_size=224):
    """
    Save WSI images with highlighted patches based on attention scores.

    Args:
        all_attn_weights (list): List of attention scores for each WSI.
        all_filenames (list): List of WSI filenames corresponding to each attention score.
        predictions (list): Model predictions (0 or 1).
        input_path (str): Path to the WSI images.
        suffix (str): Experiment suffix for saving results.
        threshold (float): Threshold for highlighting patches.
        patch_size (int): Size of each patch extracted from the WSI.
    """
    explainability_dir = f"explainability{suffix}"
    os.makedirs(explainability_dir, exist_ok=True)
    print(all_filenames)

    for i, (attn_weights, wsi_name) in enumerate(zip(all_attn_weights, all_filenames)):
        if int(predictions[i]) == 1:  # Only highlight for cancer predictions
            print(attn_weights)
            wsi_img_path = os.path.join(f"{data_path}images/", f"{wsi_name}.svs")
            mask_img_path = os.path.join(f"{data_path}patches/masks/", f"{wsi_name}.jpg")
            h5_patch_path = os.path.join(f"{data_path}patches/patches/", f"{wsi_name}.h5")

            if os.path.exists(mask_img_path) and os.path.exists(h5_patch_path) and os.path.exists(wsi_img_path):  # Ensure WSI, patches and masks exist
                mask_img = plt.imread(mask_img_path)  # Load WSI image

                # Load patch coordinates from the .h5 file
                with h5py.File(h5_patch_path, "r") as f:
                    patches = f["coords"][:]

                # Normalize attention scores
                attn_scores = attn_weights[0].flatten()
                print(attn_scores)
                attn_scores = (attn_scores - np.min(attn_scores)) / (np.max(attn_scores) - np.min(attn_scores) + 1e-8)

                # Get WSI dimensions (px)
                slide = OpenSlide(wsi_img_path)
                WSI_WIDTH, WSI_HEIGHT = slide.dimensions
                # Get mask dimensions (px)
                MASK_HEIGHT, MASK_WIDTH = mask_img.shape[:2]

                # Calculate scaling factors
                SCALE_X = MASK_WIDTH / WSI_WIDTH
                SCALE_Y = MASK_HEIGHT / WSI_HEIGHT

                # Create empty heatmap
                heatmap = np.zeros((MASK_HEIGHT, MASK_WIDTH), dtype=np.float32)

                # Overlay attention scores at correct WSI locations
                for (x, y), attn in zip(patches, attn_scores):
                    assert len(patches) == len(attn_scores), f"Mismatch: {len(patches)} patches vs {len(attn_scores)} attn scores"

                    # Adjust the coordinates and patch size
                    x, y = int(x*SCALE_X), int(y*SCALE_Y)
                    print(f"Coordinates: {x}, {y} with attention {attn}")
                    heatmap[y:y + patch_size, x:x + patch_size] += attn  # Accumulate attention
                
                print(f"For the image: {wsi_name} with dimentions: {MASK_HEIGHT}, {MASK_WIDTH}")
                
                # Normalize heatmap to [0, 1]
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                
                # Resize heatmap to match WSI dimensions
                heatmap_resized = cv2.GaussianBlur(heatmap, (5, 5), 0)  # Smoothen heatmap
                heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)

                # Blend heatmap with original WSI
                overlay = cv2.addWeighted(mask_img, 0.6, heatmap_colored, 0.4, 0)

                # Apply threshold mask for high-attention patches
                highlight_mask = heatmap_resized > threshold
                contours, _ = cv2.findContours(highlight_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)  # Green contours

                # Save the explainability image
                plt.figure(figsize=(10, 10))
                plt.imshow(overlay)
                plt.axis("off")
                plt.title(f"WSI {wsi_name} - Highlighted Patches")
                plt.savefig(os.path.join(explainability_dir, f"{wsi_name[0]}.jpg"), bbox_inches="tight", dpi=300)
                plt.close()
            else:
                print(f"Skipping {wsi_name}: WSI or patches file not found.")