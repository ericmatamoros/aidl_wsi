"""Script to extract features from patches"""
import time
import os
import argparse
import pdb
from functools import partial
from loguru import logger

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
from tqdm import tqdm

import numpy as np
import platform

from mil_wsi.CLAM  import (save_hdf5, Dataset_All_Bags, Whole_Slide_Bag_FP, get_encoder)


def compute_w_loader(output_path, loader, model, verbose = 0):
	""""
    Computes features from a PyTorch model and saves them in an HDF5 file.

    This function processes a DataLoader batch by batch, extracts features 
    using the given model, and saves them along with their coordinates.

    Args:
        output_path (str): Path to save computed features in HDF5 format.
        loader (torch.utils.data.DataLoader): DataLoader providing image batches and coordinates.
        model (torch.nn.Module): PyTorch model for feature extraction.
        device (torch.device): Device for computation ('cuda', 'mps', or 'cpu').
        verbose (int, optional): Level of feedback (0 = no output, 1 = print batch count). Defaults to 0.

    Returns:
        str: Path where the computed features are saved.
    """
	if verbose > 0:
		logger.info(f'processing a total of {len(loader)} batches'.format(len(loader)))

	mode = 'w'
	for count, data in enumerate(tqdm(loader)):
		with torch.inference_mode():	
			batch = data['img']
			coords = data['coord'].numpy().astype(np.int32)
			batch = batch.to(device, non_blocking=True)
			
			features = model(batch)
			features = features.cpu().numpy().astype(np.float32)

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--experiment_name', type = str,
					help='name of the experiment')
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--model_name', type=str, default='resnet50_trunc', choices=['resnet50_trunc', 'uni_v1', 'conch_v1'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
parser.add_argument('--num_workers', type=int, default=8)
args = parser.parse_args()

if __name__ == '__main__':
	logger.info('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	bags_dataset = Dataset_All_Bags(f"{csv_path}/{args.experiment_name}/process_list_autogen.csv")
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, f"{args.experiment_name}/", 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, f"{args.experiment_name}/", 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir,f"{args.experiment_name}/", 'pt_files'))

	model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
			
	_ = model.eval()
	model = model.to(device)
	total = len(bags_dataset)

	loader_kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if device.type in ["cuda", "mps"] else {}


	for bag_candidate_idx in tqdm(range(total)):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, f"{args.experiment_name}/", 'patches', bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		logger.info('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		logger.info(slide_id)

		if not os.path.exists(h5_file_path):
			logger.info(f'Skipping {slide_id} because {h5_file_path} does not exist')
			continue

		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			logger.info('skipped {}'.format(slide_id))
			continue 

		output_path = os.path.join(args.feat_dir,  f"{args.experiment_name}/", 'h5_files', bag_name)
		time_start = time.time()
		wsi = openslide.open_slide(slide_file_path)
		dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, 
							   		 wsi=wsi, 
									 img_transforms=img_transforms)

		loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
		output_file_path = compute_w_loader(output_path, loader = loader, model = model, verbose = 1)

		time_elapsed = time.time() - time_start
		logger.info('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))

		with h5py.File(output_file_path, "r") as file:
			features = file['features'][:]
			logger.info('features size: ', features.shape)
			logger.info('coordinates size: ', file['coords'].shape)

		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)
		torch.save(features, os.path.join(args.feat_dir,  f"{args.experiment_name}/", 'pt_files', bag_base+'.pt'))



