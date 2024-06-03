#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon June 2 2024
@author: Aobo Li
This code runs inference over the validation dataset to perform denoising task. Possible
denoising algorithms includes:

 - Moving Average [mavg]
 - Savitzky-Golay (SG) filter [savgol]
 - Fully Connected Network [fcnet]
 - Positional U-Net [punet]
 - Transformer [transformer]

For each validation file in the validation dataset, this script will produce a denoised SQUID time series,
and save it into a new file at the same location of the input validation files. For example, if the input validation file is:

/address/to/abra_validation_0000.h5

If running this inference script with trained PUNet, This script will create a new file with name and address:

/address/to/abra_validation_denoised_punet_0000.h5

Please note: if running deep learning model inference, the trained deep learning model must be present at the
same directory of this script. You can either obtain the trained deep learning model by running train.py, or download
our trained model from this address:

https://drive.google.com/drive/folders/16ORX1b2zo1_lOYYAcRBgddBuYImj0Bxs?usp=sharing

"""

import numpy as np
import argparse
from datetime import date
import torch
import torch.nn as nn
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import os
import math
from torch.utils.data import dataset
from network import PositionalUNet, TransformerModel, AE
from array2h5 import create_abra_file
from scipy.signal import savgol_filter


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Run time series denoising algorithm over the full validation dataset to produce denoised SQUID time series.")

# Output directory with default as current directory
parser.add_argument('--data_dir', '-d', type=str, default=os.path.join(os.getcwd(),"Data"), help='Directory where the training file is stored (default: current working directory).')
parser.add_argument('--denoising_model', '-m', type=str, default='punet', help='Denoising model we would like to train [mavg/savgol/fcnet/punet/transformer] (Default: punet).')

args = parser.parse_args()


NON_ML = False
#set number of training batches and input ouput size
input_size = 40000
batchsize = 25 # input_size//batchsize is the length of each time series
if args.denoising_model == "savgol" or args.denoising_model == "mavg":
	input_size = 1000000
	batchsize = 1
	NON_ML = True
	window_size = 100

output_size = input_size
ADC_CHANNEL = 256

def normalize(time_series):
    time_series = time_series[::100] #subsample a shorter TS to calculate mean and std
    return time_series.mean(), time_series.std()

def read_loader(ABRAfile):
	alltrain = np.array(ABRAfile['timeseries']['channel0001']['timeseries'])+128
	alltarget = np.array(ABRAfile['timeseries']['channel0002']['timeseries'])+128

	max_index = 2000000000
	alltrain = alltrain[:max_index].reshape( -1, batchsize, input_size)
	alltarget = alltarget[:max_index].reshape(-1, batchsize, input_size)

	return np.concatenate([alltrain,alltarget],axis=1)

def main():
	'''
	Due to the large frequency variation, we have to train different models for different frequency ranges.
	Files with higher file numbers have higher frequencies, so we train 4 models per architecture to adapt to
	different frequency ranges:
	 - Low Frequency: {Model}_0_4.pth
	 - Medium Frequency: {Model}_4_10.pth
	 - Medium-High Frequency: {Model}_10_15.pth
	 - High Frequency: {Model}_15_20.pth
	For more detail, please read appendix A of the paper
	'''
	ifile_checkpoint =  [0,4,10,15,20]
	ifile_low = ifile_checkpoint[:-1]
	ifile_high = ifile_checkpoint[1:]

	file_list = []
	for ifile in range(0,20):
		if ifile<10:
			fname = f"abra_validation_000{ifile}.h5"
		elif ifile<100:
			fname = f"abra_validation_00{ifile}.h5"
		else:
			fname = f"abra_validation_0{ifile}.h5"
		if not os.path.exists(os.path.join(args.data_dir,fname)):
			continue

		fname = os.path.join(args.data_dir,fname)

		for imodel in range(len(ifile_low)):
			low = ifile_low[imodel]
			high = ifile_high[imodel]

			if ifile >= low and (ifile < high):
				if args.denoising_model == "punet":
					model = torch.load(f'PUNet_{low}_{high}.pth',map_location=DEVICE)
					model.eval()
				elif args.denoising_model == "transformer":
					model = torch.load(f'Transformer_{low}_{high}.pth',map_location=DEVICE)
					model.eval()
				elif args.denoising_model == "fcnet":
					# There is only 1 FCNet so always load the same model
					model = torch.load(f'FCNet_{low}_{high}.pth',map_location=DEVICE)
					model.eval()
				else:
					'''
					Do nothing, this is reserved for Non-ML models
					'''
				break

		ABRAfile = h5py.File(os.path.join(args.data_dir,fname),'r')
		denoised = []
		injected = []
		train_loader = read_loader(ABRAfile)
		for i, batch in tqdm(enumerate(train_loader)):
			inputarr, targetarr = (batch[:batchsize], batch[batchsize:])

			if args.denoising_model == "mavg":
				# Size of moving average kernel
				kernel = np.ones(window_size) / window_size
				output_seq = np.convolve(inputarr.flatten(), kernel,mode="same")
			elif args.denoising_model == "savgol":
				output_seq = savgol_filter(inputarr[0].flatten(), window_size, 11)
			elif args.denoising_model == "fcnet":
				input_seq = torch.from_numpy(inputarr)
				# Forward pass
				input_seq = input_seq.float().to(DEVICE)
				output_seq = model(input_seq).detach().cpu().numpy()
			else:
				input_seq = torch.from_numpy(inputarr)
				# Forward pass
				input_seq = input_seq.long().to(DEVICE)
				output_seq = model(input_seq).argmax(dim=1).detach().cpu().numpy()
			denoised.append(np.int8(output_seq-128).flatten())
			injected.append(np.int8(targetarr-128).flatten())


		denoised = np.concatenate(denoised,axis=0)
		injected = np.concatenate(injected,axis=0)
		print(denoised, injected)
		create_abra_file(fname.replace("abra_validation", f"abra_validation_denoised_{args.denoising_model}"),denoised, injected,indexed=False)

if __name__ == "__main__":
	main()

