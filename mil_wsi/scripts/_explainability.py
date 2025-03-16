import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import h5py
from openslide import OpenSlide
from scipy.ndimage import gaussian_filter

def visualize_attention(all_attn_weights, all_filenames, predictions, data_path, files_h5_path, masks_path, suffix, threshold=0.5, patch_size=257, transformerMIL=False):
    explainability_dir = f"explainability{suffix}"
    os.makedirs(explainability_dir, exist_ok=True)
    #breakpoint()
    masks_path = "./mil_wsi/data/patches_normal/masks/"
    for i, (attn_weights, wsi_name) in enumerate(zip(all_attn_weights, all_filenames)):
        wsi_img_path = os.path.join(f"{data_path}", f"{wsi_name}.svs")
        mask_img_path = os.path.join(f"{masks_path}", f"{wsi_name}.jpg")
        file_h5_path = os.path.join(f"{files_h5_path}", f"{wsi_name}.h5")

        if os.path.exists(mask_img_path) and os.path.exists(file_h5_path) and os.path.exists(wsi_img_path):
            mask_img = plt.imread(mask_img_path)
            with h5py.File(file_h5_path, "r") as f:
                patches = f["coords"][:]

            attn_scores = attn_weights[0].flatten() if not transformerMIL else attn_weights.mean(dim=(0, 1)).mean(dim=1)

            # Plot histogram before normalization
            plt.figure(figsize=(10, 4))
            plt.hist(attn_scores, bins=50, alpha=0.75, color='blue', label='Before Normalization')
            plt.xlabel('Attention Score')
            plt.ylabel('Frequency')
            plt.title(f'Attention Scores Distribution Before Normalization - {wsi_name}')
            plt.legend()
            plt.savefig(os.path.join(explainability_dir, f"{wsi_name}_hist_before.jpg"), bbox_inches="tight", dpi=300)
            plt.close()

            min_val, max_val = np.percentile(attn_scores, 1), np.percentile(attn_scores, 99)
            attn_scores = (attn_scores - min_val) / (max_val - min_val + 1e-10)
            attn_scores = np.clip(attn_scores, 0, 1)

            # Plot histogram after normalization
            plt.figure(figsize=(10, 4))
            plt.hist(attn_scores, bins=50, alpha=0.75, color='red', label='After Normalization')
            plt.xlabel('Attention Score')
            plt.ylabel('Frequency')
            plt.title(f'Attention Scores Distribution After Normalization - {wsi_name}')
            plt.legend()
            plt.savefig(os.path.join(explainability_dir, f"{wsi_name}_hist_after.jpg"), bbox_inches="tight", dpi=300)
            plt.close()

            slide = OpenSlide(wsi_img_path)
            WSI_WIDTH, WSI_HEIGHT = slide.dimensions
            MASK_HEIGHT, MASK_WIDTH = mask_img.shape[:2]
            SCALE_X, SCALE_Y = MASK_WIDTH / WSI_WIDTH, MASK_HEIGHT / WSI_HEIGHT
            heatmap = np.zeros((MASK_HEIGHT, MASK_WIDTH), dtype=np.float32)

            for (x, y), attn in zip(patches, attn_scores):
                x, y = int(x * SCALE_X), int(y * SCALE_Y)
                heatmap[y:y + int(patch_size * SCALE_Y), x:x + int(patch_size * SCALE_X)] += attn

            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)
            
            # Apply Gaussian smoothing before coloring
            heatmap_smoothed = gaussian_filter(heatmap, sigma=4)
            
            heatmap_colored = cv2.applyColorMap((heatmap_smoothed * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            overlay = cv2.addWeighted(mask_img, 0.4, heatmap_colored, 0.6, 0)
            fig, ax = plt.subplots(figsize=(10, 10))
            im = ax.imshow(overlay)
            ax.axis("off")
            ax.set_title(f"WSI {wsi_name} - Highlighted Patches")

            # Adjust layout and set colorbar height proportional to image height
            fig.subplots_adjust(right=0.85)
            bbox = ax.get_position()
            cbar_height = bbox.y1 - bbox.y0  # Height of the image
            cbar_ax = fig.add_axes([0.87, bbox.y0, 0.03, cbar_height])  # Align with image height
            cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='jet'), cax=cbar_ax)
            cbar.set_label("Attention Score", rotation=270, labelpad=15)

            prediction_label = str(predictions[i])
            save_dir = os.path.join(explainability_dir, prediction_label)
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"{wsi_name}.jpg"), bbox_inches="tight", dpi=300)
            plt.close()
        else:
            print(f"Path not found: {mask_img_path}")
