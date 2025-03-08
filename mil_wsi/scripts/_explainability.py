import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import h5py
from openslide import OpenSlide


def visualize_attention(all_attn_weights, all_filenames, predictions, data_path, suffix, threshold=0.5, patch_size=257):
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
        print(attn_weights)
        wsi_img_path = os.path.join(f"{data_path}images/", f"{wsi_name}.svs")
        mask_img_path = os.path.join(f"{data_path}patches/masks/", f"{wsi_name}.jpg")
        h5_patch_path = os.path.join(f"{data_path}patches/patches/", f"{wsi_name}.h5")

        if os.path.exists(mask_img_path) and os.path.exists(h5_patch_path) and os.path.exists(wsi_img_path):  # Ensure WSI, patches and masks exist
            mask_img = plt.imread(mask_img_path)  # Load WSI image

            # Load patch coordinates from the .h5 file
            with h5py.File(h5_patch_path, "r") as f:
                patches = f["coords"][:]

            attn_scores = attn_weights[0].flatten()

            # Evitar que todo sea 0 si la variabilidad es baja
            min_val, max_val = np.percentile(attn_scores, 1), np.percentile(attn_scores, 99)  # Recortar valores extremos inferiores

            attn_scores = (attn_scores - min_val) / (max_val - min_val + 1e-10)
            attn_scores = np.clip(attn_scores, 0, 1)  # Asegurar que los valores estén entre [0,1]

            # Get WSI dimensions (px)
            slide = OpenSlide(wsi_img_path)
            WSI_WIDTH, WSI_HEIGHT = slide.dimensions
            # Get mask dimensions (px)
            MASK_HEIGHT, MASK_WIDTH = mask_img.shape[:2]

            # Calculate scaling factors
            SCALE_X, SCALE_Y = MASK_WIDTH / WSI_WIDTH, MASK_HEIGHT / WSI_HEIGHT

            # Create empty heatmap
            heatmap = np.zeros((MASK_HEIGHT, MASK_WIDTH), dtype=np.float32)

            # Overlay attention scores at correct WSI locations
            for (x, y), attn in zip(patches, attn_scores):
                assert len(patches) == len(attn_scores), f"Mismatch: {len(patches)} patches vs {len(attn_scores)} attn scores"

                # Adjust the coordinates and patch size
                x, y = int(x*SCALE_X), int(y*SCALE_Y)
                heatmap[y:y + int(patch_size*SCALE_X), x:x + int(patch_size*SCALE_Y)] += attn  # Accumulate attention

            
            # Normalize heatmap to [0, 1]
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)
                
            
            # Aply color map
            heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET) # Esta en BGR
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            # Blend heatmap with original WSI
            overlay = cv2.addWeighted(mask_img, 0.4, heatmap_colored, 0.6, 0)


            # **Adding a colorbar (legend)**
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(overlay)
            ax.axis("off")
            ax.set_title(f"WSI {wsi_name} - Highlighted Patches")

            # Create an additional colorbar legend
            cax = fig.add_axes([0.85, 0.2, 0.03, 0.6])  # [left, bottom, width, height] in figure coordinates
            heatmap_for_legend = np.linspace(0, 1, 256).reshape(256, 1)  # Fake heatmap for the colorbar
            cbar = ax.figure.colorbar(plt.cm.ScalarMappable(cmap='jet'), cax=cax)
            cbar.set_label("Attention Score", rotation=270, labelpad=15)
            
            # Save in diferent directiories depending on the prediction
            if int(predictions[i]) == 1:
                save_dir = f"{explainability_dir}/cancer"
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f"{wsi_name}.jpg"), bbox_inches="tight", dpi=300)
                plt.close()
            else:
                save_dir = f"{explainability_dir}/no_cancer"
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f"{wsi_name}.jpg"), bbox_inches="tight", dpi=300)
                plt.close()