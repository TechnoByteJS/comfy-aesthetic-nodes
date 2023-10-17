from PIL import Image
from os.path import join
import ImageReward as RM
import clip
import datetime
import folder_paths
import io
import json
import math
import numpy as np
import os
import pytorch_lightning as pl
import re
import socket
import statistics
import sys
import time
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import comfy.sd
import comfy.utils

# Aesthetic Scoring Node
folder_paths.folder_names_and_paths["aesthetic"] = ([os.path.join(folder_paths.models_dir,"aesthetic")], folder_paths.supported_pt_extensions)


class MLP(pl.LightningModule):
	def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
		super().__init__()
		self.input_size = input_size
		self.xcol = xcol
		self.ycol = ycol
		self.layers = nn.Sequential(
			nn.Linear(self.input_size, 1024),
			#nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(1024, 128),
			#nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(128, 64),
			#nn.ReLU(),
			nn.Dropout(0.1),
			nn.Linear(64, 16),
			#nn.ReLU(),
			nn.Linear(16, 1)
		)
	def forward(self, x):
		return self.layers(x)
	def training_step(self, batch, batch_idx):
			x = batch[self.xcol]
			y = batch[self.ycol].reshape(-1, 1)
			x_hat = self.layers(x)
			loss = F.mse_loss(x_hat, y)
			return loss
	def validation_step(self, batch, batch_idx):
		x = batch[self.xcol]
		y = batch[self.ycol].reshape(-1, 1)
		x_hat = self.layers(x)
		loss = F.mse_loss(x_hat, y)
		return loss
	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		return optimizer
def normalized(a, axis=-1, order=2):
	import numpy as np  # pylint: disable=import-outside-toplevel
	l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
	l2[l2 == 0] = 1
	return a / np.expand_dims(l2, axis)


class AestheticNode_Scoring:
	def __init__(self):
		pass

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"model_name": (folder_paths.get_filename_list("aesthetic"), {"multiline": False, "default": "chadscorer.pth"}),
				"image": ("IMAGE",),
				}
		}

	RETURN_TYPES = ("NUMBER","IMAGE")
	FUNCTION = "calc_score"
	CATEGORY = "Scoring"

	def calc_score(self, model_name, image):
		m_path = folder_paths.folder_names_and_paths["aesthetic"][0]
		m_path2 = os.path.join(m_path[0], model_name)
		model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
		s = torch.load(m_path2, map_location=torch.device('cpu'))
		model.load_state_dict(s)
		# model.to("cuda")
		model.eval()
		device = "cpu" 
		model2, preprocess = clip.load("ViT-L/14", device=device)  # RN50x64
		tensor_image = image[0]
		img = (tensor_image * 255).to(torch.uint8).numpy()
		pil_image = Image.fromarray(img, mode='RGB')
		image2 = preprocess(pil_image).unsqueeze(0).to(device)
		with torch.no_grad():
				image_features = model2.encode_image(image2)
				im_emb_arr = normalized(image_features.detach().numpy())
				prediction = model(torch.from_numpy(im_emb_arr).to(torch.device("cpu")).type(torch.FloatTensor))
				final_prediction = round(float(prediction[0]), 2)
		del model
		return (final_prediction,)

# Image Reward Scoring Node
class AestheticNode_ImageReward:
	def __init__(self):
		self.model = None

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"model": ("STRING", {"multiline": False, "default": "ImageReward-v1.0"}),
				"prompt": ("STRING", {"multiline": True, "forceInput": True}),
				"images": ("IMAGE",),
			},
		}

	RETURN_TYPES = ("FLOAT", "STRING", "FLOAT", "STRING")
	RETURN_NAMES = ("SCORE_FLOAT", "SCORE_STRING", "VALUE_FLOAT", "VALUE_STRING")

	CATEGORY = "Scoring"

	FUNCTION = "process_images"

	def process_images(self, model, prompt, images,): #rounded):
		if self.model is None:
			self.model = RM.load(model)

		score = 0.0
		for image in images:
			# convert to PIL image
			i = 255.0 * image.cpu().numpy()
			img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
			score += self.model.score(prompt, [img])
		score /= len(images)
		# assume std dev follows normal distribution curve
		valuescale = 0.5 * (1 + math.erf(score / math.sqrt(2))) * 10  # *10 to get a value between -10
		return (score, str(score), valuescale, str(valuescale))

# CREDITS
#----------------------------------------------
# Endless Sea of Stars Custom Node Collection
# https://github.com/tusharbhutt/Endless-Nodes
#----------------------------------------------
#
