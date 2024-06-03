"""
Created on Mon June 2 2024
@author: Jessica Fry, Aobo Li
This code analyzes the abra_validation_denoised_* file produced by inference.py script to
produce Benchmark 1: Denoising Score.

This code is parallelized with python concurrent.future command. With 8 cores, the fine scan takes
roughly 30 minutes to calculate, while the coarse scan takes roughly 3 minutes

"""
import numpy as np
import h5py as h5
from matplotlib import pyplot as plt
import matplotlib as mpl
import logging
import argparse
import os
import gc
from tqdm import tqdm
import concurrent.futures
import math

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times"
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.ERROR)

def GetOneSecPSD(file_path, files, ch, start = 0):
    file_list = []

    if type(files) == list:
        for f in files:
            file_list.append(os.path.join(file_path,f))
    elif files.endswith(".h5"):
        file_list = [os.path.join(file_path,files)]
        logging.info('Getting data from h5 file')
    else:
        logging.error('Not acceptable data format!')
    N = 10000000
    psd_sum = np.zeros(int(N/2))
    interfile_data = []
    total_psds = 0
    N_sec_file = 200
    
    fileNum = start // N_sec_file
    startIndex = N * (start % N_sec_file)

    file = file_list[fileNum]
    logging.info("Opening file "+file)
    with h5.File(file) as h5f:
        if ch == 1:
            data = h5f['timeseries']['channel0001']['timeseries'][startIndex:startIndex+N]
        elif ch == 2:
            data = h5f['timeseries']['channel0002']['timeseries'][startIndex:startIndex+N]
        else:
            logging.error("Incorrect channel number. Choose 1 or 2")
        volt_range = (h5f['timeseries']['channel0001']).attrs['voltage_range_mV']
        logging.info("Retrieved data from h5.")

        scaling = np.float32(volt_range/(2*128.0))
        TS = np.array(data, dtype= np.float32)*scaling
        logging.info("Retrieved time series.")
            
        dt = 1.0/(h5.File(file)['timeseries']['channel0001']).attrs['sampling_frequency']

        psd_chunk = dt/N*(abs(np.fft.rfft(TS.reshape(len(TS)//N,N)))**2).sum(0)[1:]
        freq_array = np.linspace(0,5*1e6,int(N/2))
    del data, TS, dt
    gc.collect()
    return freq_array, psd_chunk

def findPeak(pwr):
    peakdiff = pwr[1:-1]-pwr[:-2]-pwr[2:]
    peakIndex = int(np.where(peakdiff==np.amax(peakdiff))[0][0])+1
    return(peakIndex)
    
def getSNR(freq, pwr, target = 0):
    if target == 0:
        center_id = findPeak(pwr)
    else:
        center_id = int(np.where(freq==target)[0][0])
    sig_range = 1
    noise_range = 50
    signal = np.sum(pwr[center_id-sig_range:center_id+sig_range+1])
    noise = np.sum(pwr[center_id-noise_range:center_id+noise_range+1])-signal
    return [signal/noise, freq[center_id]]

def process_iteration(i, path, file, coarse):
    start_index = i * 10 if coarse else i
    # Process channel 2
    freq_sg, psd_sg = GetOneSecPSD(path, file, ch=2, start=start_index)
    snr_sg, center_freq = getSNR(freq_sg, psd_sg)
    
    # Process channel 1
    freq_squid, psd_squid = GetOneSecPSD(path, file, ch=1, start=start_index)
    snr_squid = getSNR(freq_squid, psd_squid, center_freq)[0]
    return i, snr_sg, snr_squid

############################# Example Usage ###################################
# score = calculateBenchmark('/data/', ['file1.h5', 'file2.h5'], coarse = True)
#
# NOTE: the HDF5 files must be in the abra format. Use array2h5.py script
#       to convert any array(s) to an abra format. If using a list of files,
#       each file must be of length 200 seconds (2000000000 length array) as 
#       prescribed by the array2h5.py script. If using a single .h5 file, 
#       it can be any length.
###############################################################################
def calculateBenchmark(path, file, args):
    #n = 4199
    if type(file) == list:
        n = 200 * len(file)
    elif file.endswith(".h5"):
        with h5py.File(os.path.join(path,file), 'r') as file:
            dataset = file['/timeseries/channel0001/timeseries']
            length = dataset.shape[0]
            n = length // 10000000 #use sampling rate to get number of seconds
    else:
        print("Incorrect file format")
    if args.coarse:
        n = int(n/10)
    snr_squid = np.zeros(shape=(n,))
    snr_sg = np.zeros(shape=(n,))
    if args.parallel:
        # Initialize the executor
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # Create a list of tasks
            tasks = [executor.submit(process_iteration, i, path, file, args.coarse) for i in range(n)]
            
            # Process the results as they become available
            for future in tqdm(concurrent.futures.as_completed(tasks), total=n):
                i, result_snr_sg, result_snr_squid = future.result(timeout=3)
                snr_sg[i] = result_snr_sg
                snr_squid[i] = result_snr_squid
    else:
        for i in tqdm(range(n)):
            i, result_snr_sg, result_snr_squid = process_iteration(i, path, file, args.coarse)
            snr_sg[i] = result_snr_sg
            snr_squid[i] = result_snr_squid
    snr_sg = snr_sg/(np.amax(snr_sg))
    score = np.round(np.sum(np.multiply(snr_sg, snr_squid))/snr_squid.size,decimals=2) + 1e-10
    return math.log(score,5.27)

parser = argparse.ArgumentParser(description="Calculate Benchmark 1: Denoising Score using the produced files from inference.py")
parser.add_argument('--data_dir', '-d', type=str, default=os.getcwd(), help='Directory where the training file is stored (default: current working directory).')
parser.add_argument('--denoising_model', '-m', type=str, default='punet', help='Denoising model we would like to train [none/savgol/mavg/punet/transformer] (Default: punet).')
parser.add_argument('-c', '--coarse', action='store_true', help='Running a coarse scan instead of fine scan to compute denoising score.')
parser.add_argument('-p', '--parallel', action='store_true', help='Running the denoising score calculation with multiprocessing.')
parser.add_argument('-w', '--num_workers', type=int,  help='maximum number of workers for parallel processing, default: 32', default=32)
args = parser.parse_args()
args = parser.parse_args()

file_list = []
for i in range(20):
    if i<10:
        fname = f"abra_validation_denoised_{args.denoising_model}_000{i}.h5"
    elif i<100:
        fname = f"abra_validation_denoised_{args.denoising_model}_00{i}.h5"
    else:
        fname = f"abra_validation_denoised_{args.denoising_model}_0{i}.h5"
    if args.denoising_model == "none":
        fname = fname.replace("_denoised_none", "")
    if os.path.exists(os.path.join(args.data_dir,fname)):
        file_list.append(fname)
print(file_list)
score = calculateBenchmark(args.data_dir+"/", file_list, args)
is_coarse = "Coarse" if args.coarse else "Fine"
print(f"{is_coarse} Denoising Score for Model {args.denoising_model}: {score}")
