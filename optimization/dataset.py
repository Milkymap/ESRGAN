import torch as th 
import numpy as np 

from glob import glob 
from os import path 

from PIL import Image 
from torch.utils.data import Dataset 
from torchvision import transforms as T
from libraries.strategies import cv2th, read_image

class Source(Dataset):
	def __init__(self, root_path, target_shape=(256, 256), upscale_factor=4, augmentation_rate=0.7):
		super(Source, self).__init__()
		W, H = target_shape  # (W, H)
		self.image_paths = sorted(glob(path.join(root_path, '*.jpg')))[:4000]
		self.augmentation_rate = augmentation_rate
		self.build_lr = T.Compose([
			T.Resize((W // upscale_factor, H // upscale_factor), interpolation=T.InterpolationMode.BICUBIC),
			T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
		])
		self.build_hr = T.Compose([
			T.Resize((W, H)),
			T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
		])

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, index):
		original = read_image(self.image_paths[index], by='th').float()
		lr_image = self.build_lr(original)  
		hr_image = self.build_hr(original)
		return lr_image, hr_image