import streamlit as st
import torch
import numpy as np
import os
import cv2
import h5py
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from openslide import OpenSlide
from torch.utils.data import DataLoader
from mil_wsi.interfaces import predict_attention_mil, MILBagDataset, AttentionMILMLP
from mil_wsi.CLAM import WholeSlideImage, StitchCoords, initialize_df, save_hdf5, Dataset_All_Bags, Whole_Slide_Bag_Get_FP, get_encoder
import time
from tqdm import tqdm

from mil_wsi.scripts.create_patches import seg_and_patch
from mil_wsi.scripts.extract_features import compute_w_loader

def process_wsi(pt_file_path, target_csv, model, device, batch_size=1):
    """Loads the extracted features and generates predictions."""
    target = pd.read_csv(target_csv)
    target['filename'] = target['slide'].str.replace('.svs', '', regex=False)
    num_classes = len(np.unique(target['target'].values))

    if 'wsi label' not in target.columns:
        raise Exception("Not a valid real category column named 'wsi label' found in targets file (.csv)")
    classes = target[['wsi label', 'target']].drop_duplicates()

    dataset_path = pt_file_path  # <- Make sure this is the correct path
    dataset_files = os.listdir(dataset_path)  # List all .pt files
    dataset_path = dataset_path.split("pt_files")[0]

    dataset = MILBagDataset(dataset_path, dataset_files, target)  # Remove extra "pt_files/"
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    predictions, attn_weights, bag_ids = predict_attention_mil(model, dataloader, device, num_classes)
    predictions = predictions.cpu().numpy().round().astype(int)

    preds_class = classes[classes['target'] == predictions[0]]['wsi label'].values[0]
    return predictions, attn_weights, bag_ids, preds_class

def generate_heatmap(wsi_path, attn_weights, bag_ids, patch_size=257):
    """
    Generate a heatmap overlay on the WSI image using attention scores.

    Args:
        wsi_path (str): Path to the Whole Slide Image (.svs).
        attn_weights (list): Attention scores for each patch.
        bag_ids (list): Patch coordinates (x, y).
        patch_size (int): Size of each patch in pixels.

    Returns:
        fig (matplotlib.figure.Figure): The figure containing the heatmap overlay.
    """

    # Load WSI and get dimensions
    slide = OpenSlide(wsi_path)
    WSI_WIDTH, WSI_HEIGHT = slide.dimensions

    # Create empty heatmap
    heatmap = np.zeros((WSI_HEIGHT, WSI_WIDTH), dtype=np.float32)

    # Normalize attention scores
    attn_scores = attn_weights[0].flatten()
    min_val, max_val = np.percentile(attn_scores, 1), np.percentile(attn_scores, 99)
    attn_scores = (attn_scores - min_val) / (max_val - min_val + 1e-10)
    attn_scores = np.clip(attn_scores, 0, 1)

    # Overlay attention scores on the heatmap
    for (x, y), attn in zip(bag_ids, attn_scores):
        x, y = int(x), int(y)
        heatmap[y:y + patch_size, x:x + patch_size] += attn  # Accumulate attention

    # Normalize heatmap to range [0,1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)

    # Apply color map
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Convert WSI to image format (downsample for visualization)
    wsi_thumbnail = np.array(slide.get_thumbnail((1024, int(1024 * WSI_HEIGHT / WSI_WIDTH))))
    wsi_thumbnail_resized = cv2.resize(wsi_thumbnail, (WSI_WIDTH, WSI_HEIGHT))

    # Blend heatmap with original WSI
    overlay = cv2.addWeighted(wsi_thumbnail_resized, 0.5, heatmap_colored, 0.5, 0)

    # Create Matplotlib figure for Streamlit display
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(overlay)
    ax.axis("off")
    ax.set_title("Attention Heatmap")

    # Create an additional colorbar legend
    cax = fig.add_axes([0.85, 0.2, 0.03, 0.6])  # [left, bottom, width, height] in figure coordinates
    cbar = ax.figure.colorbar(plt.cm.ScalarMappable(cmap='jet'), cax=cax)
    cbar.set_label("Attention Score", rotation=270, labelpad=15)

    return fig  # Returning the figure instead of showing it

# Streamlit UI
st.title("WSI MIL Prediction App")

uploaded_file = st.file_uploader("Upload a Whole Slide Image (.svs)", type=["svs"])
uploaded_target = st.file_uploader("Upload Target CSV File", type=["csv"])
uploaded_model = st.file_uploader("Upload Trained Model (.pth)", type=["pth"])

temp_dir = tempfile.TemporaryDirectory()

temp_wsi_dir = os.path.join(temp_dir.name, "wsi")
os.makedirs(temp_wsi_dir, exist_ok=True)

temp_patch_dir = os.path.join(temp_dir.name, "patches")
temp_mask_dir = os.path.join(temp_dir.name, "masks")
temp_stitch_dir = os.path.join(temp_dir.name, "stitches")
temp_patch_mask_dir = os.path.join(temp_dir.name, "patches_on_mask")
temp_feat_dir = os.path.join(temp_dir.name, "features")
temp_pt_dir = os.path.join(temp_dir.name, "pt_files")

os.makedirs(temp_patch_dir, exist_ok=True)
os.makedirs(temp_mask_dir, exist_ok=True)
os.makedirs(temp_stitch_dir, exist_ok=True)
os.makedirs(temp_patch_mask_dir, exist_ok=True)
os.makedirs(temp_feat_dir, exist_ok=True)
os.makedirs(temp_pt_dir, exist_ok=True)

if uploaded_file and uploaded_target and uploaded_model:
    temp_wsi_path = os.path.join(temp_wsi_dir, uploaded_file.name)
    with open(temp_wsi_path, "wb") as f:
        f.write(uploaded_file.read())
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_csv:
        temp_csv.write(uploaded_target.read())
        target_csv_path = temp_csv.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as temp_model:
        temp_model.write(uploaded_model.read())
        model_path = temp_model.name
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    
    st.write("Generating patches...")
    directories = {
        'source': temp_wsi_dir, 
        'save_dir': temp_dir.name,
        'patch_save_dir': temp_patch_dir, 
        'mask_save_dir': temp_mask_dir, 
        'stitch_save_dir': temp_stitch_dir,
        'mask_on_patch_save_dir': temp_patch_mask_dir
    }
    
    seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                  'keep_ids': 'none', 'exclude_ids': 'none'}
    filter_params = {'a_t': 100, 'a_h': 16, 'max_n_holes': 8}
    vis_params = {'vis_level': -1, 'line_thickness': 250}
    patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}
    
    parameters = {'seg_params': seg_params,
                  'filter_params': filter_params,
                  'patch_params': patch_params,
                  'vis_params': vis_params}
    
    seg_and_patch(**directories, **parameters, patch_size=256, step_size=256, 
                  seg=True, use_default_params=False, save_mask=True, 
                  stitch=True, patch=True, patch_on_mask=False, process_list=None, auto_skip=True)
    
    st.write("Extracting features...")
    bags_dataset = Dataset_All_Bags(os.path.join(temp_dir.name, 'process_list_autogen.csv'))
    model_extraction, img_transforms = get_encoder("resnet50_trunc", target_img_size=224)
    model_extraction.eval().to(device)
    
    for bag_candidate_idx in tqdm(range(len(bags_dataset))):
        slide_id = bags_dataset[bag_candidate_idx]
        bag_name = f"{slide_id.split('.svs')[0]}.h5"
        h5_file_path = os.path.join(temp_patch_dir, bag_name)
        slide_file_path = temp_wsi_path
        
        if not os.path.exists(h5_file_path):
            continue

        dataset = Whole_Slide_Bag_Get_FP(file_path=h5_file_path, slide_path=slide_file_path, img_transforms=img_transforms)        
        loader = DataLoader(dataset, batch_size=256, num_workers=4, pin_memory=True)
        output_file = os.path.join(temp_feat_dir, bag_name)
        
        output_file_path = compute_w_loader(output_file, loader, model_extraction)
        
        with h5py.File(output_file_path, "r") as file:
            features = file['features'][:]
        
        features = torch.from_numpy(features)
        bag_base, _ = os.path.splitext(bag_name)
        pt_file_path = os.path.join(temp_pt_dir, f"{slide_id.split('.svs')[0]}.pt")
        torch.save(features, pt_file_path)
    
    st.write("Processing WSI...")
    predictions, attn_weights, bag_ids, pred_class = process_wsi(temp_pt_dir, target_csv_path, model, device)
    
    st.write(f"Predicted label: {pred_class}")

    patch_size=257
    wsi_img_path = os.path.join(f"{temp_wsi_dir}", f"{bag_ids[0]}.svs")
    mask_img_path = os.path.join(f"{temp_mask_dir}", f"{bag_ids[0]}.jpg")
    h5_patch_path = os.path.join(f"{temp_patch_dir}", f"{bag_ids[0]}.h5")

    wsi_name = bag_ids[0]

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


        st.pyplot(fig)