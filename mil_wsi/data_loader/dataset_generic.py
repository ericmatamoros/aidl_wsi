import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
from scipy import stats
from .utils import get_split_loader, generate_split, nth
from torch.utils.data import Dataset
import h5py


class Generic_WSI_Classification_Dataset(Dataset):
	def __init__(self,
		csv_path = 'dataset_csv/ccrcc_clean.csv',
		shuffle = False, 
		seed = 7, 
		print_info = True,
		label_dict = {},
		filter_dict = {},
		ignore=[],
		patient_strat=False,
		label_col = None,
		patient_voting = 'max',
		):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			shuffle (boolean): Whether to shuffle
			seed (int): random seed for shuffling the data
			print_info (boolean): Whether to print a summary of the dataset
			label_dict (dict): Dictionary with key, value pairs for converting str labels to int
			ignore (list): List containing class labels to ignore
		"""
		self.label_dict = label_dict
		self.num_classes = len(set(self.label_dict.values()))
		self.seed = seed
		self.print_info = print_info
		self.patient_strat = patient_strat
		self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
		self.data_dir = None
		if not label_col:
			label_col = 'label'
		self.label_col = label_col

		print("Calling __init__ from Class Generic_WSI_Classification_Dataset...")

		slide_data = pd.read_csv(csv_path)
		#The filter_dict parameter and filter_df function allows you to filter the dataset by keeping only rows that match specific values in certain columns
		# If you want to load only a subset of the dataset, based on column values you would pass the varieble. Ex: filter_dict = {"tissue_type": ["skin"]}
		slide_data = self.filter_df(slide_data, filter_dict)
		slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col)

		###shuffle data
		if shuffle:
			np.random.seed(seed)
			np.random.shuffle(slide_data)

		self.slide_data = slide_data

		# We eliminated patient_data_prep() and cls_ids_prep()since we are not going to use patient information

		#if print_info:
		#	self.summarize()


	@staticmethod
	def df_prep(data, label_dict, ignore, label_col):
		"""
		1-Renaming the label column (if necessary).
		2-Removing rows that contain labels you want to ignore.
		3-Converting textual labels to numeric labels (based on label_dict).			
		"""
		print("Calling df_prep()...")
		if label_col != 'label':
			#data['label'] = data[label_col].copy() # Creates a new column called label if it doesn't exist
			data.rename(columns={label_col: "label"}, inplace=True) # If label_col is Not "label", Rename It to "label"

		mask = data['label'].isin(ignore) # ignore is a list of labels that should not be included in the dataset. If ignore=["negative"], then all rows with "negative" labels are removed.
		data = data[~mask]
		data.reset_index(drop=True, inplace=True) # if rows are removed we re-assign sequental index numbers
		
		# Convert labels to numbers
		data['label'] = data['label'].map(label_dict)  # ✅ Direct mapping instead of looping

		#for i in data.index:
		#	key = data.loc[i, 'label'] #Goes row by row, looks at the label (key), and replaces it with its numeric value from label_dict
		#	data.at[i, 'label'] = label_dict[key]

		print (data)

		return data

	#The filter_dict parameter allows you to filter the dataset by keeping only rows that match specific values in certain columns.
	def filter_df(self, df, filter_dict={}):
		print("Calling filter_df()...")
		if len(filter_dict) > 0:
			# Creates an array of True values with the same length as the dataset. This mask will be used to decide which rows to keep (True) and which to remove (False).
			filter_mask = np.full(len(df), True, bool) 
			# assert 'label' not in filter_dict.keys()
			for key, val in filter_dict.items():
				print(key)
				print(val)
				mask = df[key].isin(val)
				filter_mask = np.logical_and(filter_mask, mask)
			df = df[filter_mask]
			print(df)
		else:
			print("Filter dict is empty,  no filtering happens")
		return df

	def __len__(self):
		if self.patient_strat:
			return len(self.patient_data['case_id'])

		else:
			return len(self.slide_data)

	def get_list(self, ids):
		return self.slide_data['slide_id'][ids]

	def getlabel(self, ids):
		return self.slide_data['label'][ids]

	def __getitem__(self, idx):
		return None


class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
	def __init__(self,
		data_dir, 
		**kwargs):
	
		super(Generic_MIL_Dataset, self).__init__(**kwargs) #This means that before doing anything else, Python will first execute the __init__ function of the parent class (Generic_WSI_Classification_Dataset).
		self.data_dir = data_dir
		self.use_h5 = False

	def load_from_h5(self, toggle):
		self.use_h5 = toggle

	def __getitem__(self, idx):
		slide_id = self.slide_data['slide_id'][idx]
		label = self.slide_data['label'][idx]
		if type(self.data_dir) == dict:
			source = self.slide_data['source'][idx]
			data_dir = self.data_dir[source]
		else:
			data_dir = self.data_dir

		if not self.use_h5:
			if self.data_dir:
				full_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id))
				features = torch.load(full_path, weights_only=True)
				return features, label, slide_id
			
			else:
				return slide_id, label

		else:
			full_path = os.path.join(data_dir,'h5_files','{}.h5'.format(slide_id))
			with h5py.File(full_path,'r') as hdf5_file:
				features = hdf5_file['features'][:]
				coords = hdf5_file['coords'][:]

			features = torch.from_numpy(features)
			return features, label, coords, slide_id
