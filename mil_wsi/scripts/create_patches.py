"""Script to create patches"""
from mil_wsi.CLAM import (WholeSlideImage, StitchCoords, initialize_df)
# other imports
import os
import numpy as np
import time
import argparse
import pdb
import h5py
import cv2
from openslide import OpenSlide
import pandas as pd
from tqdm import tqdm

from loguru import logger

def stitching(file_path, wsi_object, downscale = 64):
	"""
    Stitches patches into a heatmap representation of the WSI.

    Args:
        file_path (str): Path to the patch coordinate file.
        wsi_object: Whole Slide Image (WSI) object.
        downscale (int, optional): Downscaling factor for stitching. Defaults to 64.

    Returns:
        np.ndarray: Heatmap representation of the stitched patches.
        float: Time taken for the stitching process.
    """
	start = time.time()
	heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0,0,0), alpha=-1, draw_grid=False)
	total_time = time.time() - start
	
	return heatmap, total_time

def segment(WSI_object, seg_params = None, filter_params = None, mask_file = None):
	"""
    Performs tissue segmentation on a WSI.

    Args:
        WSI_object: Whole Slide Image (WSI) object.
        seg_params (dict, optional): Segmentation parameters.
        filter_params (dict, optional): Filtering parameters for tissue segmentation.
        mask_file (str, optional): Path to a precomputed segmentation mask.

    Returns:
        WSI_object: Updated WSI object after segmentation.
        float: Time taken for the segmentation process.
    """
	### Start Seg Timer
	start_time = time.time()
	# Use segmentation file
	if mask_file is not None:
		WSI_object.initSegmentation(mask_file)
	# Segment	
	else:
		WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

	### Stop Seg Timers
	seg_time_elapsed = time.time() - start_time   
	return WSI_object, seg_time_elapsed

def patching(WSI_object, **kwargs):
	"""
    Extracts patches from the segmented WSI.

    Args:
        WSI_object: Whole Slide Image (WSI) object.
        **kwargs: Additional parameters for patch extraction.

    Returns:
        str: File path to the generated patches.
        float: Time taken for the patch extraction process.
    """
	### Start Patch Timer
	start_time = time.time()

	# Patch
	file_path = WSI_object.process_contours(**kwargs)


	### Stop Patch Timer
	patch_time_elapsed = time.time() - start_time
	return file_path, patch_time_elapsed


def get_wsi_dimensions(wsi_path):
    """
    Retrieves the original dimensions of a Whole Slide Image (WSI).

    Args:
        wsi_path (str): Path to the WSI file.

    Returns:
        tuple: (width, height) of the WSI.
    """
    """Gets the dimensions of the original WSI."""
    slide = OpenSlide(wsi_path)
    return slide.dimensions  # Returns (width, height)

def get_target_dimensions(target_path):
    """
    Retrieves the dimensions of a visualization file (e.g., mask or stitched image).

    Args:
        target_path (str): Path to the target visualization file.

    Returns:
        tuple: (width, height) of the target file.

    Raises:
        FileNotFoundError: If the file is not found or cannot be read.
    """
    """Gets the dimensions of the visualization file (masks or stitches)."""
    target_image = cv2.imread(target_path)
    if target_image is None:
        raise FileNotFoundError(f"Target file not found: {target_path}")
    return target_image.shape[1], target_image.shape[0]  # Returns (width, height)

def draw_patches_on_target(target_path, coords, output_path, scale_x, scale_y, patch_size):
    """
    Draws rectangular patches on a visualization file (e.g., mask or stitched image).

    Args:
        target_path (str): Path to the target visualization file.
        coords (list of tuples): List of (x, y) coordinates of patches.
        output_path (str): Path to save the image with drawn patches.
        scale_x (float): Scaling factor for x-coordinates.
        scale_y (float): Scaling factor for y-coordinates.
        patch_size (int): Size of each patch in pixels.

    Saves:
        - Image with drawn rectangles highlighting patch locations.
    """
    """Draws rectangles on the visualization file (masks or stitches)."""
    # Load the visualization file
    target_image = cv2.imread(target_path)
    if target_image is None:
        logger.info(f"Error loading target: {target_path}")
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


def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir, mask_on_patch_save_dir,
				  patch_size = 256, step_size = 256, 
				  seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'},
				  filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}, 
				  vis_params = {'vis_level': -1, 'line_thickness': 500},
				  patch_params = {'use_padding': True, 'contour_fn': 'four_pt'},
				  patch_level = 0,
				  use_default_params = False, 
				  seg = False, save_mask = True, 
				  stitch= False, 
				  patch = False, 
				  patch_on_mask = False,
				  auto_skip=True, process_list = None):
	"""
    Performs segmentation, patch extraction, and visualization for Whole Slide Images (WSIs).

    This function processes WSIs by:
    1. Segmenting tissue regions using specified segmentation parameters.
    2. Extracting patches based on detected regions.
    3. Optionally stitching extracted patches into a heatmap representation.
    4. Saving mask visualizations and overlaying patch locations on masks.

    Args:
        source (str): Path to the directory containing WSI files.
        save_dir (str): Directory where the process list CSV will be saved.
        patch_save_dir (str): Directory to save extracted patches.
        mask_save_dir (str): Directory to save segmentation masks.
        stitch_save_dir (str): Directory to save stitched heatmaps.
        mask_on_patch_save_dir (str): Directory to save masks with patch overlays.
        patch_size (int, optional): Size of each extracted patch in pixels. Defaults to 256.
        step_size (int, optional): Step size for patch extraction. Defaults to 256.
        seg_params (dict, optional): Parameters for tissue segmentation.
        filter_params (dict, optional): Parameters for filtering tissue regions.
        vis_params (dict, optional): Parameters for mask visualization.
        patch_params (dict, optional): Parameters for patch extraction.
        patch_level (int, optional): Image pyramid level for patch extraction. Defaults to 0.
        use_default_params (bool, optional): If True, use default parameters for all processes. Defaults to False.
        seg (bool, optional): If True, perform tissue segmentation. Defaults to False.
        save_mask (bool, optional): If True, save segmentation masks. Defaults to True.
        stitch (bool, optional): If True, generate stitched heatmaps. Defaults to False.
        patch (bool, optional): If True, extract patches. Defaults to False.
        patch_on_mask (bool, optional): If True, overlay extracted patch locations on masks. Defaults to False.
        auto_skip (bool, optional): If True, skip processing if patches already exist. Defaults to True.
        process_list (str, optional): Path to a CSV file specifying which slides to process. If None, all slides are processed.

    Returns:
        float: Average segmentation time per slide.
        float: Average patch extraction time per slide.
    """

	slides = sorted(os.listdir(source))
	slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide))]
	slides = [slide for slide in slides if all(ext not in slide for ext in ['git', 'csv', 'xlsx', '.DS_Store'])] # TO BE IMPROVED

	if process_list is None:
		df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)
	
	else:
		df = pd.read_csv(process_list)
		df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

	mask = df['process'] == 1
	process_stack = df[mask]

	total = len(process_stack)

	legacy_support = 'a' in df.keys()
	if legacy_support:
		logger.info('detected legacy segmentation csv file, legacy support enabled')
		df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
		'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
		'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
		'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
		'contour_fn': np.full((len(df)), patch_params['contour_fn'])})

	seg_times = 0.
	patch_times = 0.
	stitch_times = 0.

	for i in tqdm(range(total)):
		df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
		idx = process_stack.index[i]
		slide = process_stack.loc[idx, 'slide_id']
		logger.info("\n\nprogress: {:.2f}, {}/{}".format(i/total, i, total))
		logger.info('processing {}'.format(slide))
		
		df.loc[idx, 'process'] = 0
		slide_id, _ = os.path.splitext(slide)

		if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
			logger.info('{} already exist in destination location, skipped'.format(slide_id))
			df.loc[idx, 'status'] = 'already_exist'
			continue

		# Inialize WSI
		full_path = os.path.join(source, slide)
		WSI_object = WholeSlideImage(full_path)

		if use_default_params:
			current_vis_params = vis_params.copy()
			current_filter_params = filter_params.copy()
			current_seg_params = seg_params.copy()
			current_patch_params = patch_params.copy()
			
		else:
			current_vis_params = {}
			current_filter_params = {}
			current_seg_params = {}
			current_patch_params = {}


			for key in vis_params.keys():
				if legacy_support and key == 'vis_level':
					df.loc[idx, key] = -1
				current_vis_params.update({key: df.loc[idx, key]})

			for key in filter_params.keys():
				if legacy_support and key == 'a_t':
					old_area = df.loc[idx, 'a']
					seg_level = df.loc[idx, 'seg_level']
					scale = WSI_object.level_downsamples[seg_level]
					adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
					current_filter_params.update({key: adjusted_area})
					df.loc[idx, key] = adjusted_area
				current_filter_params.update({key: df.loc[idx, key]})

			for key in seg_params.keys():
				if legacy_support and key == 'seg_level':
					df.loc[idx, key] = -1
				current_seg_params.update({key: df.loc[idx, key]})

			for key in patch_params.keys():
				current_patch_params.update({key: df.loc[idx, key]})

		if current_vis_params['vis_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_vis_params['vis_level'] = 0
			
			else:	
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_vis_params['vis_level'] = best_level

		if current_seg_params['seg_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_seg_params['seg_level'] = 0
			
			else:
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_seg_params['seg_level'] = best_level

		keep_ids = str(current_seg_params['keep_ids'])
		if keep_ids != 'none' and len(keep_ids) > 0:
			str_ids = current_seg_params['keep_ids']
			current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['keep_ids'] = []

		exclude_ids = str(current_seg_params['exclude_ids'])
		if exclude_ids != 'none' and len(exclude_ids) > 0:
			str_ids = current_seg_params['exclude_ids']
			current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['exclude_ids'] = []

		w, h = WSI_object.level_dim[current_seg_params['seg_level']] 
		if w * h > 1e8:
			logger.info('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
			df.loc[idx, 'status'] = 'failed_seg'
			continue

		df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
		df.loc[idx, 'seg_level'] = current_seg_params['seg_level']


		seg_time_elapsed = -1
		if seg:
			WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params) 

		if save_mask:
			mask = WSI_object.visWSI(**current_vis_params)
			mask_path = os.path.join(mask_save_dir, slide_id+'.jpg')
			mask.save(mask_path)

		patch_time_elapsed = -1 # Default time
		if patch:
			current_patch_params.update({'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size, 
										 'save_path': patch_save_dir})
			file_path, patch_time_elapsed = patching(WSI_object = WSI_object,  **current_patch_params,)
		
		stitch_time_elapsed = -1
		if stitch:
			file_path = os.path.join(patch_save_dir, slide_id+'.h5')
			if os.path.isfile(file_path):
				heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=64)
				stitch_path = os.path.join(stitch_save_dir, slide_id+'.jpg')
				heatmap.save(stitch_path)

		if patch_on_mask:
			# Automatically get dimensions (px)
			WSI_WIDTH, WSI_HEIGHT = get_wsi_dimensions(full_path)
			TARGET_WIDTH, TARGET_HEIGHT = get_target_dimensions(os.path.join(mask_save_dir, slide_id+'.jpg'))

			# Calculate scaling factors
			SCALE_X = TARGET_WIDTH / WSI_WIDTH
			SCALE_Y = TARGET_HEIGHT / WSI_HEIGHT

			with h5py.File( os.path.join(patch_save_dir, slide_id+'.h5'), 'r') as f:
				coords = f['coords'][:]
			
			draw_patches_on_target(
				os.path.join(mask_save_dir, slide_id+'.jpg'), 
				coords, 
				os.path.join(mask_on_patch_save_dir, slide_id+'.jpg'), SCALE_X, SCALE_Y, patch_size)
						

		logger.info("segmentation took {} seconds".format(seg_time_elapsed))
		logger.info("patching took {} seconds".format(patch_time_elapsed))
		logger.info("stitching took {} seconds".format(stitch_time_elapsed))
		df.loc[idx, 'status'] = 'processed'

		seg_times += seg_time_elapsed
		patch_times += patch_time_elapsed
		stitch_times += stitch_time_elapsed

	seg_times /= total
	patch_times /= total
	stitch_times /= total

	df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
	logger.info("average segmentation time in s per slide: {}".format(seg_times))
	logger.info("average patching time in s per slide: {}".format(patch_times))
	logger.info("average stiching time in s per slide: {}".format(stitch_times))
		
	return seg_times, patch_times

parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--source', type = str,
					help='path to folder containing raw wsi image files')
parser.add_argument('--experiment_name', type = str,
					help='name of the experiment')
parser.add_argument('--step_size', type = int, default=256,
					help='step_size')
parser.add_argument('--patch_size', type = int, default=256,
					help='patch_size')
parser.add_argument('--patch', default=False, action = 'store_true', help='store H5 with the coordinates of the patches for a WSI')
parser.add_argument('--seg', default=False, action='store_true')
parser.add_argument('--stitch', default=False,  action='store_true', help='store filtered WSI post-masking')
parser.add_argument('--patch_on_mask', default=False,action='store_true', help='store WSI image with overlapping patches')
parser.add_argument('--no_auto_skip', default=True, action='store_false')
parser.add_argument('--save_dir', type = str,
					help='directory to save processed data')
parser.add_argument('--preset', default=None, type=str,
					help='predefined profile of default segmentation and filter parameters (.csv)')
parser.add_argument('--patch_level', type=int, default=0, 
					help='downsample level at which to patch')
parser.add_argument('--process_list',  type = str, default=None,
					help='name of list of images to process with parameters (.csv)')

if __name__ == '__main__':
	args = parser.parse_args()

	patch_save_dir = os.path.join(args.save_dir, f"{args.experiment_name}/", 'patches')
	mask_save_dir = os.path.join(args.save_dir, f"{args.experiment_name}/", 'masks')
	stitch_save_dir = os.path.join(args.save_dir, f"{args.experiment_name}/", 'stitches')
	mask_on_patch_save_dir = os.path.join(args.save_dir, f"{args.experiment_name}/", 'patches_on_mask')

	if args.process_list:
		process_list = os.path.join(args.save_dir, args.process_list)

	else:
		process_list = None

	logger.info('source: ', args.source)
	logger.info('patch_save_dir: ', patch_save_dir)
	logger.info('mask_save_dir: ', mask_save_dir)
	logger.info('stitch_save_dir: ', stitch_save_dir)
	logger.info('mask_on_patch: ', mask_on_patch_save_dir)
	
	directories = {'source': args.source, 
				   'save_dir': os.path.join(args.save_dir, f"{args.experiment_name}/"),
				   'patch_save_dir': patch_save_dir, 
				   'mask_save_dir' : mask_save_dir, 
				   'stitch_save_dir': stitch_save_dir,
				   'mask_on_patch_save_dir': mask_on_patch_save_dir} 

	for key, val in directories.items():
		logger.info("{} : {}".format(key, val))
		if key not in ['source']:
			os.makedirs(val, exist_ok=True)

	seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'}
	filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}
	vis_params = {'vis_level': -1, 'line_thickness': 250}
	patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

	if args.preset:
		preset_df = pd.read_csv(os.path.join('presets', args.preset))
		for key in seg_params.keys():
			seg_params[key] = preset_df.loc[0, key]

		for key in filter_params.keys():
			filter_params[key] = preset_df.loc[0, key]

		for key in vis_params.keys():
			vis_params[key] = preset_df.loc[0, key]

		for key in patch_params.keys():
			patch_params[key] = preset_df.loc[0, key]
	
	parameters = {'seg_params': seg_params,
				  'filter_params': filter_params,
	 			  'patch_params': patch_params,
				  'vis_params': vis_params}

	logger.info(parameters)

	seg_times, patch_times = seg_and_patch(**directories, **parameters,
											patch_size = args.patch_size, step_size=args.step_size, 
											seg = args.seg,  use_default_params=False, save_mask = True, 
											stitch= args.stitch,
											patch_level=args.patch_level, patch = args.patch,
											patch_on_mask = args.patch_on_mask,
											process_list = process_list, 
											auto_skip=args.no_auto_skip)
