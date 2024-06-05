import wget
import argparse
import os
import urllib
import subprocess
import scitokens
import string
import sys
import numpy as np
import argparse
from datetime import date
import torch
import torch.nn as nn
import h5py as h5
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import os
import requests
import math
from torch.utils.data import dataset
from network import PositionalUNet, FocalLoss1D, TransformerModel, AE
from array2h5 import create_abra_file
import time
import logging


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description="Train time series denoising model over the full training dataset to produce result")

# Output directory with default as current directory, change the default to path to directory where unprocessed science data is located
parser.add_argument('--data_dir', '-d', type=str, default=os.path.join(os.getcwd(),"Data"), help='Directory where the science file is stored (default: current working directory).')
#change the defult type to 'punet', 'fcnet', or 'transformer to run different types of models
parser.add_argument('--denoising_model', '-m', type=str, default='punet', help='Denoising model we would like to train [punet/transformer/fcnet] (Default: punet).')

args = parser.parse_args()

#set number of training batches and input ouput size
input_size = 40000
if args.denoising_model == "transformer":
    input_size = 20000
    batchsize = 25 # input_size//batchsize is the length of each time series
batchsize = 25 # input_size//batchsize is the length of each time series
output_size = input_size
ADC_CHANNEL = 256

def read_loader(ABRAfile):
    alltrain = np.array(ABRAfile['timeseries']['channel0001']['timeseries'])+128
    #change the max_index to length of science data TS series 
    max_index = 4010000000
    alltrain = alltrain[:max_index].reshape( -1, batchsize, input_size)

    return alltrain


class cd:
   '''
   Context manager for changing the current working directory
   '''
   def __init__(self, newPath):
      self.newPath = newPath

   def __enter__(self):
      self.savedPath = os.getcwd()
      os.chdir(self.newPath)

   def __exit__(self, etype, value, traceback):
      os.chdir(self.savedPath)

def download_type(folder, file_prefix, i, path=os.getcwd()):
	# Download training data file
	if i < 10:
		fname = file_prefix+"000%d.h5"%(i)
	elif i<100:
		fname = file_prefix+"00%d.h5"%(i)
	else:
		fname = file_prefix+"0%d.h5"%(i)
	url = prefix%(folder+"/"+fname)
	print(url)
	if os.path.exists(fname):
		return fname
	try:
		wget.download(url,out = path)
	except urllib.error.HTTPError as err:
		print(f"File {fname} not downloaded! Error message {err}")
		return None
	except urllib.error.URLError as err2:
		print(f"File {fname} not downloaded! Error message {err2}, retry downkloading with a different cache.")
		subprocess.run(["wget", url.replace("https://osdf-director.osg-htc.org", "https://osg-sunnyvale-stashcache.t2.ucsd.edu:8443"), "--no-check-certificate", "-P", path])


	return fname

def check_range(min_val, max_val):
    """Defines a range check function for argparse."""
    class Range:
        def __init__(self, min_val, max_val):
            self.min_val = min_val
            self.max_val = max_val

        def __call__(self, value):
            value_int = int(value)
            if value_int < self.min_val or value_int > self.max_val:
                raise argparse.ArgumentTypeError(f"Value must be between {self.min_val} and {self.max_val}")
            return value_int
    return Range(min_val, max_val)



def process_denoised(denoised, note = ''):
    '''
    connecting and saving denoised data
    '''
    denoised = np.concatenate(denoised,axis=0)
    #set the name of the output file
    outname = 'denoised_' + note + '.h5'
    create_abra_file(file_name = outname, array1=denoised)


def main(args):
    #change range to process specific science data files
    
    for ifile in range(0,207):
        #change to prefix of downloaded science data files 
        if ifile<10:
            fname = f"MagnetOn_80mVADC_ChAIntegAxion_RF30kOhm_10kHzHPF_5MHzLPF_000{ifile}.h5"
        elif ifile<100:
            fname = f"MagnetOn_80mVADC_ChAIntegAxion_RF30kOhm_10kHzHPF_5MHzLPF_00{ifile}.h5"
        else:
            fname = f"MagnetOn_80mVADC_ChAIntegAxion_RF30kOhm_10kHzHPF_5MHzLPF_0{ifile}.h5"
            
        if not os.path.exists(os.path.join(args.data_dir,fname)):
            print(f"Science run {fname} not found in {args.data_dir}!")
            continue
        
        if args.denoising_model == "punet":
            mname = "PUNet"
        elif args.denoising_model == "transformer":
            mname = "Transformer"    
        elif args.denoising_model == "fcnet":
            mname = "FCNet"

        model1 = torch.load(f'{mname}_0_4.pth',map_location=DEVICE)
        model2 = torch.load(f'{mname}_4_10.pth',map_location=DEVICE)
        model3 = torch.load(f'{mname}_10_15.pth',map_location=DEVICE)
        model4 = torch.load(f'{mname}_15_20.pth',map_location=DEVICE)

        model1.eval()
        model2.eval()
        model3.eval()
        model4.eval()
        ABRAfile = h5.File(os.path.join(args.data_dir,fname))
        denoised1 = []
        denoised2 = []
        denoised3 = []
        denoised4 = []
        train_loader = read_loader(ABRAfile)
        # os.remove(fname)
		# del ABRAfile
        gc.collect()
        for i, batch in tqdm(enumerate(train_loader)):
            input_seq = torch.from_numpy(batch)
            #forward pass
            input_seq = input_seq.int().to(DEVICE)
            if args.denoising_model == "fcnet":
                output_seq1 = model1(input_seq)
                output_seq2 = model2(input_seq)
                output_seq3 = model3(input_seq)
                output_seq4 = model4(input_seq)
                
            else:
                output_seq1 = model1(input_seq).argmax(dim=1)
                output_seq2 = model2(input_seq).argmax(dim=1)
                output_seq3 = model3(input_seq).argmax(dim=1)
                output_seq4 = model4(input_seq).argmax(dim=1)

            denoised1.append(np.int8(output_seq1.detach().cpu().numpy().flatten()-128))
            denoised2.append(np.int8(output_seq2.detach().cpu().numpy().flatten()-128))
            denoised3.append(np.int8(output_seq3.detach().cpu().numpy().flatten()-128))
            denoised4.append(np.int8(output_seq4.detach().cpu().numpy().flatten()-128))

		# These are 4 version of the denoised time series with parameters trained on different frequency ranges, change the note var to specify in saved file name
        process_denoised(denoised1, note = mname+'_0_4pth_file'+str(ifile))
        process_denoised(denoised2, note = mname+'_4_10pth_file'+str(ifile))
        process_denoised(denoised3, note = mname+'_10_15pth_file'+str(ifile))
        process_denoised(denoised4, note = mname+'_15_20pth_file'+str(ifile))


if __name__ == "__main__":
		main(args)
