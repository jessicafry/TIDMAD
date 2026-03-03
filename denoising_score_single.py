#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon June 2 2024
@author: Jessica Fry, Aobo Li

Modified for Single File Denoising Score calculation within the Sandbox.
"""
import numpy as np
import h5py as h5
import h5py
import logging
import argparse
import os
import gc
from tqdm import tqdm
import concurrent.futures
import math

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.ERROR)

def GetOneSecPSD(file_path, files, ch, start = 0):
    file_list = []

    if type(files) == list:
        for f in files:
            file_list.append(os.path.join(file_path,f))
    elif files.endswith(".h5"):
        file_list = [os.path.join(file_path,files)]
    else:
        logging.error('Not acceptable data format!')
        
    N = 10000000
    fileNum = start // 200 # Original logic assumes 200s per file
    startIndex = N * (start % 200)

    file = file_list[fileNum]
    with h5.File(file, 'r') as h5f:
        if ch == 1:
            data = h5f['timeseries']['channel0001']['timeseries'][startIndex:startIndex+N]
        elif ch == 2:
            data = h5f['timeseries']['channel0002']['timeseries'][startIndex:startIndex+N]
        
        volt_range = h5f['timeseries']['channel0001'].attrs['voltage_range_mV']
        sampling_freq = h5f['timeseries']['channel0001'].attrs['sampling_frequency']

        scaling = np.float32(volt_range/(2*128.0))
        TS = np.array(data, dtype= np.float32)*scaling
        dt = 1.0 / sampling_freq

        psd_chunk = dt/N*(abs(np.fft.rfft(TS))**2)[1:]
        freq_array = np.linspace(0, sampling_freq/2, int(N/2))
        
    del data, TS
    gc.collect()
    return freq_array, psd_chunk

def findPeak(pwr):
    peakdiff = pwr[1:-1]-pwr[:-2]-pwr[2:]
    peakIndex = int(np.where(peakdiff==np.amax(peakdiff))[0][0])+1
    return peakIndex
    
def getSNR(freq, pwr, target = 0):
    if target == 0:
        center_id = findPeak(pwr)
    else:
        center_id = int(np.where(freq==target)[0][0])
    sig_range = 1
    noise_range = 50
    signal = np.sum(pwr[center_id-sig_range:center_id+sig_range+1])
    noise = np.sum(pwr[center_id-noise_range:center_id+noise_range+1])-signal
    if noise <= 0:
        noise = 1e-5
    return [signal/noise, freq[center_id]]

def process_iteration(i, path, file, coarse):
    start_index = i * 10 if coarse else i
    freq_sg, psd_sg = GetOneSecPSD(path, file, ch=2, start=start_index)
    snr_sg, center_freq = getSNR(freq_sg, psd_sg)
    
    freq_squid, psd_squid = GetOneSecPSD(path, file, ch=1, start=start_index)
    snr_squid = getSNR(freq_squid, psd_squid, center_freq)[0]
    return i, snr_sg, snr_squid

def calculateBenchmark(path, file, args):
    if type(file) == list:
        file_to_open = os.path.join(path, file[0])
    else:
        file_to_open = os.path.join(path, file)

    with h5py.File(file_to_open, 'r') as f:
        dataset = f['/timeseries/channel0001/timeseries']
        length = dataset.shape[0]
        n = length // 10000000 

    if args.coarse:
        n = int(n/10)
        
    snr_squid = np.zeros(shape=(n,))
    snr_sg = np.zeros(shape=(n,))
    
    if args.parallel:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            tasks = [executor.submit(process_iteration, i, path, file, args.coarse) for i in range(n)]
            for future in tqdm(concurrent.futures.as_completed(tasks), total=n, desc="Calculating SNR"):
                i, result_snr_sg, result_snr_squid = future.result()
                snr_sg[i] = result_snr_sg
                snr_squid[i] = result_snr_squid
    else:
        for i in tqdm(range(n)):
            i, result_snr_sg, result_snr_squid = process_iteration(i, path, file, args.coarse)
            snr_sg[i] = result_snr_sg
            snr_squid[i] = result_snr_squid
            
    snr_sg = snr_sg/(np.amax(snr_sg) if np.amax(snr_sg) != 0 else 1.0)
    score = np.round(np.sum(np.multiply(snr_sg, snr_squid))/snr_squid.size, decimals=2) + 1e-10
    return math.log(score, 5.27)

# ==========================================
# Parser Definitions (Kept exactly as requested)
# ==========================================
parser = argparse.ArgumentParser(description="Calculate Benchmark 1: Denoising Score using the produced files from inference.py")
parser.add_argument('--data_dir', '-d', type=str, default="/home/klz/Data/TIDMAD/", help='Directory where the training file is stored.')
parser.add_argument('--denoising_model', '-m', type=str, default='punet', help='Denoising model [none/savgol/mavg/punet/transformer].')
parser.add_argument('-c', '--coarse', action='store_true', help='Running a coarse scan instead of fine scan.')
parser.add_argument('-p', '--parallel', action='store_true', help='Running with multiprocessing.')
parser.add_argument('-n', '--num_workers', type=int, default=8, help='maximum number of workers, default: 8')
parser.add_argument('--delete_files', action='store_true', help='Delete validation files after running.')
parser.add_argument('-w', '--weak', action='store_true', help='validate on weak version (10mV)')
parser.add_argument('--file_index', '-i', type=int, default=0, help="Index for the single file to be loaded")

args = parser.parse_args()

# Single file targeting logic based on file_index
if args.weak:
    actual_index = args.file_index + 20
else:
    actual_index = args.file_index

# Construct the filename consistent with your naming convention
fname = f"abra_validation_denoised_{args.denoising_model}_{str(actual_index).zfill(4)}.h5"
if args.denoising_model == "none":
    fname = f"abra_validation_{str(actual_index).zfill(4)}.h5"

full_path = os.path.join(args.data_dir, fname)

if os.path.exists(full_path):
    print(f"Calculating score for: {fname}")
    score = calculateBenchmark(args.data_dir, [fname], args)
    is_coarse = "Coarse" if args.coarse else "Fine"
    print(f"\n{fname} {is_coarse} Denoising Score: {score:.4f}")
    
    if args.delete_files:
        os.remove(full_path)
        print(f"Deleted: {full_path}")
else:
    print(f"Error: File not found at {full_path}")