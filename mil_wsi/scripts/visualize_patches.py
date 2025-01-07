import h5py
import cv2
import os
import matplotlib.pyplot as plt
from openslide import OpenSlide

# Directories
PATCHES_DIR = "data/patches/patches"  # Path where the .h5 files are located
WSI_DIR = "data/wsi"  # Path where the original WSI images are located
OUTPUT_DIR = "data/patches/visualized_patches"  # Folder to save the results

# Visualization target: can be 'masks' or 'stitches'
VISUALIZATION_TARGET = "masks"  # Change to "stitches" to work with that folder

# Dynamic directory based on the visualization target
TARGET_DIR = f"data/patches/{VISUALIZATION_TARGET}"  # Folder for masks or stitches

# Patch parameters
PATCH_SIZE = 256  # Size of the patches

# Create the output directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_wsi_dimensions(wsi_path):
    """Gets the dimensions of the original WSI."""
    slide = OpenSlide(wsi_path)
    return slide.dimensions  # Returns (width, height)

def get_target_dimensions(target_path):
    """Gets the dimensions of the visualization file (masks or stitches)."""
    target_image = cv2.imread(target_path)
    if target_image is None:
        raise FileNotFoundError(f"Target file not found: {target_path}")
    return target_image.shape[1], target_image.shape[0]  # Returns (width, height)

def draw_patches_on_target(target_path, coords, output_path, scale_x, scale_y, patch_size=PATCH_SIZE):
    """Draws rectangles on the visualization file (masks or stitches)."""
    # Load the visualization file
    target_image = cv2.imread(target_path)
    if target_image is None:
        print(f"Error loading target: {target_path}")
        return
    
    # Adjust the coordinates and patch size
    adjusted_patch_size_x = int(patch_size * scale_x)
    adjusted_patch_size_y = int(patch_size * scale_y)

    # Draw a rectangle for each adjusted coordinate
    for x, y in coords:
        top_left = (int(x * scale_x), int(y * scale_y))
        bottom_right = (int((x * scale_x) + adjusted_patch_size_x), int((y * scale_y) + adjusted_patch_size_y))
        color = (0, 255, 255) 
        thickness = 1
        cv2.rectangle(target_image, top_left, bottom_right, color, thickness)
    
    # Save the image with the rectangles
    cv2.imwrite(output_path, target_image)


# Process all .h5 files and corresponding files in the selected folder
for h5_file in os.listdir(PATCHES_DIR):
    if not h5_file.endswith('.h5'):
        continue
    
    # Read coordinates from the .h5 file
    h5_path = os.path.join(PATCHES_DIR, h5_file)
    with h5py.File(h5_path, 'r') as f:
        coords = f['coords'][:]
        print(f"Found {len(coords)} patches in {h5_file}.")
    
    # Find the original WSI and its corresponding visualization file
    wsi_filename = h5_file.replace('.h5', '.svs')  
    wsi_path = os.path.join(WSI_DIR, wsi_filename)
    target_filename = h5_file.replace('.h5', '.jpg')  
    target_path = os.path.join(TARGET_DIR, target_filename)
    
    if not os.path.exists(wsi_path):
        print(f"WSI not found for {h5_file}: {wsi_path}")
        continue
    if not os.path.exists(target_path):
        print(f"Target file not found for {h5_file}: {target_path}")
        continue
    
    # Automatically get dimensions (px)
    WSI_WIDTH, WSI_HEIGHT = get_wsi_dimensions(wsi_path)
    TARGET_WIDTH, TARGET_HEIGHT = get_target_dimensions(target_path)

    # Calculate scaling factors
    SCALE_X = TARGET_WIDTH / WSI_WIDTH
    SCALE_Y = TARGET_HEIGHT / WSI_HEIGHT

    print(f"{h5_file}: WSI={WSI_WIDTH}x{WSI_HEIGHT}, Target={TARGET_WIDTH}x{TARGET_HEIGHT}")

    # Output the visualization file with rectangles
    output_path = os.path.join(OUTPUT_DIR, f"{os.path.basename(target_filename)}")
    
    # Draw rectangles on the visualization file
    draw_patches_on_target(target_path, coords, output_path, SCALE_X, SCALE_Y)
