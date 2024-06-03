#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 2024
@author: Aobo Li, Hope Fu

This script trains deep learning model over the training dataset, possible architecture includes:
 - Fully Connected Network [fcnet]
 - Positional U-Net [punet]
 - Transformer [transformer]
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
from network import PositionalUNet, FocalLoss1D, TransformerModel, AE

# SQUID = h5py.File(SQUIDname,'r')
# SG = h5py.File(SGname, 'r')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Train time series denoising model over the full training dataset to produce result")

# Output directory with default as current directory
parser.add_argument('--data_dir', '-d', type=str, default=os.getcwd(), help='Directory where the training file is stored (default: current working directory).')
parser.add_argument('--denoising_model', '-m', type=str, default='punet', help='Denoising model we would like to train [fcnet/punet/transformer] (Default: punet).')
parser.add_argument('-f', '--force', action='store_true', help='Directly proceed to download without asking the confirming question.')

args = parser.parse_args()

#set the size of segmentations for deep learning models
input_size = 40000
if args.denoising_model == "transformer":
	input_size = 20000 # transformer model requires additional GPU memories, so we reduce segment size by 50%
sample_size = 50 #Randomly sample 2% of the time series to train model
batchsize = 1
output_size = input_size
ADC_CHANNEL = 256

def normalize(time_series):
    time_series = time_series[::100] #subsample a shorter TS to calculate mean and std
    return time_series.mean(), time_series.std()

def read_loader(ABRAfile):
	alltrain = np.array(ABRAfile['timeseries']['channel0001']['timeseries'])+128
	alltarget = np.array(ABRAfile['timeseries']['channel0002']['timeseries'])+128

	max_index = 2000000000
	alltrain = alltrain[:max_index].reshape( -1,sample_size, batchsize, input_size)
	alltarget = alltarget[:max_index].reshape(-1,sample_size, batchsize, input_size)
	random_index = np.random.randint(sample_size)

	return np.concatenate([alltrain[:,random_index],alltarget[:,random_index]],axis=1)

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
ifile_checkpoint = [0,4,10,15,20]

file_list = []
for ifile in range(20):
    if ifile<10:
        fname = f"abra_training_000{ifile}.h5"
    elif ifile<100:
        fname = f"abra_training_00{ifile}.h5"
    else:
        fname = f"abra_training_0{ifile}.h5"
    if not os.path.exists(os.path.join(args.data_dir,fname)):
        continue

    if ifile in ifile_checkpoint:
        # Initialize a new model at the beginning of new checkpoint
        if args.denoising_model == "punet":
            model = PositionalUNet().to(DEVICE)
            criterion = FocalLoss1D().to(DEVICE)
        elif args.denoising_model == "transformer":
            model = TransformerModel().to(DEVICE)
            criterion = FocalLoss1D().to(DEVICE)
        elif args.denoising_model == "fcnet":
            model = AE(input_size).to(DEVICE)
            criterion = nn.SmoothL1Loss().to(DEVICE)
        else:
            raise ValueError

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)


    # Read file
    ABRAfile = h5py.File(os.path.join(args.data_dir,fname),'r')

    # Start training
    train_loader = read_loader(ABRAfile)
    np.random.shuffle(train_loader)
    for i, batch in enumerate(train_loader):
        inputarr, targetarr = (batch[:batchsize], batch[batchsize:])
        input_seq = torch.from_numpy(inputarr)
        target_seq = torch.from_numpy(targetarr)
        randind = np.random.randint(batchsize)
        input_seq = input_seq[randind].unsqueeze(0).float().to(DEVICE)
        target_seq = target_seq[randind].unsqueeze(0).float().to(DEVICE)
            


        # Forward pass
        if not args.denoising_model == "fcnet":
            input_seq = input_seq.int()
            target_seq = target_seq.long()
        
    
        output_seq = model(input_seq)

        # Calculate the loss
        loss = criterion(output_seq, target_seq)

        # Backward pass and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss every 100 batches
        if i % 500 == 50:
            print('Epoch: {} | Batch: {} | Loss: {}'.format(ifile, i, loss.item()))
            plt.plot(np.arange(input_size), input_seq[0].detach().cpu().numpy().flatten(), label = 'input',alpha=0.5, lw=1)
            plt.plot(np.arange(input_size), target_seq[0].detach().cpu().numpy().flatten(), label = 'target',alpha=0.5, lw=1)
            if args.denoising_model != "fcnet":
                # If the model is not FCNet, the model accomplish a segmentation task with 256 classes per time step
                output_seq = output_seq.argmax(dim=1)
            plt.plot(np.arange(input_size), output_seq[0].detach().cpu().numpy().flatten(), label = 'output',alpha=0.5, lw=1)
            plt.legend()
            plt.savefig("denoise_sample.pdf",dpi=100)
            plt.cla()
            plt.clf()
            plt.close()
    del ABRAfile, train_loader
    gc.collect()

    # Save and delete current model at the end of current checkpoint
    if (ifile+1) in ifile_checkpoint:
        prev_file = ifile_checkpoint[ifile_checkpoint.index(ifile+1)-1]
        if args.denoising_model == "punet":
            torch.save(model, f'PUNet_{prev_file}_{ifile+1}.pth')
        elif args.denoising_model == "fcnet":
            torch.save(model, f'FCNet_{prev_file}_{ifile+1}.pth')
        elif args.denoising_model == "transformer":
            torch.save(model, f'Transformer_{prev_file}_{ifile+1}.pth')
        del model, criterion, optimizer
        torch.cuda.empty_cache()