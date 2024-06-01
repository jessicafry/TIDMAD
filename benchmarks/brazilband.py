########################################################## Example Usage ###########################################################
# From terminal:
#
# python brazilband.py [path to input files] [input files (.txt)(.npy)] [output file name (no extention)] --level coarse --v
#
#####################################################################################################################################


import numpy as np
import pandas as pd
import h5py as h5
from matplotlib import pyplot as plt
import matplotlib as mpl
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = 'serif'
mpl.use('Agg')
import logging
import analysis
from iminuit import Minuit
from scipy import stats
from scipy.interpolate import interp1d
import argparse
import ast
import os

c = 299792.458 # km/s
h = 4.135592569185604e-15 #eV/Hz
v0 = 220/c # unitless dispersion veloctiy 
vObs = 232/c # unitless boost velocity
conv = 0.19383487875847072e-13 #conversion factor from psd to calibrated g_agg 

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.WARNING)

def ReadFileFromList(file_path, file_list):
    file_df = pd.read_csv(file_path + file_list)
    file_number = len(file_df)
    file_list = []
    for i in range (0,file_number):
        file_list.append(file_path + file_df['list.txt'][i])
    return file_list

def GetAveragePSDfromTS(file_path, files, ch, start = 0, N=1000000, length_sec = 0, plot = False):
    file_list = []

    if type(files) == list:
        for f in files:
            file_list.append(file_path + f)
    else:
        logging.error('Not acceptable data format!')
    
    psd_sum = np.zeros(int(N/2))
    interfile_data = []
    total_psds = 0
    
    for file in file_list:
        logging.info("Opening file "+file)
        if ch == 1:
            data = h5.File(file)['timeseries']['channel0001']
        elif ch == 2:
            data = h5.File(file)['timeseries']['channel0002']
        else:
            logging.error("Incorrect channel number. Choose 1 or 2")
        volt_range = data.attrs['voltage_range_mV']
        logging.info("Retrieved data from h5.")

        scaling = np.float32(volt_range/(2*128.0))
        TS = np.array(data['timeseries'], dtype= np.float32)*scaling
        logging.info("Retrieved time series.")
        
        if len(interfile_data) > 0:
            logging.info('There is remaining data from the previous file.')
            np.append(TS, interfile_data)

        dt = 1.0/data.attrs['sampling_frequency']
        
        if(length_sec == 0):
            num_full_chunks = np.floor(len(TS)/N).astype(int)
        else:
            num_full_chunks = np.floor((length_sec/dt)/N).astype(int)
            if np.floor(len(TS)/N).astype(int) < num_full_chunks:
                num_full_chunks = np.floor(len(TS)/N).astype(int)
                logging.warning("You inputted a TS length greater out of bounds for the file inputted.")
            
        logging.info('There are '+str(num_full_chunks)+ ' chunks in the file')
        
        for psdnum in range(0, num_full_chunks):
            logging.info('Starting '+str(psdnum)+ '/'+str(num_full_chunks)+' chunk')
            ts_chunk = TS[int(start+psdnum*N):int(start+(psdnum+1)*N)]

            psd_chunk = dt/N*(abs(np.fft.rfft(ts_chunk.reshape(len(ts_chunk)//N,N)))**2).sum(0)[1:]
            
            logging.info('Produced PSD')
            psd_sum = np.add(psd_sum, psd_chunk)
            total_psds += 1
            
            del psd_chunk
            del ts_chunk
        if(num_full_chunks > len(TS)/N):
            interfile_data = TS[startnum_full_chunks*N:]
            logging.info('There is a leftover chunk of length '+ len(interfile_data))

    freq_array = np.linspace(0,5*1e6,int(N/2))
    psd_ave = psd_sum/total_psds

    if plot:
        logging.info("Begin plotting")
        fig = plt.figure(figsize=(12,5))
        plt.plot(freq_array, psd_ave)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel(r'$mV^2/Hz$')
        plt.yscale('log')
        plt.xscale('log')
        plt.grid()
        plt.savefig('../plots/test.png')
    
    return freq_array, psd_ave

def GetSearchFreqArray(lower = 99000, upper=2000000, n_points=10000000, abra=False):
    if abra:
        return np.geomspace(99000,2000000, 1000)
    return np.geomspace(lower, upper, n_points)
           
def GetFreqBins(freq_array):
    freq = np.zeros((2,len(freq_array)))
    delta = freq_array*5.5e-6
    freq[0] = freq_array - (delta*0.2)
    freq[1] = freq_array + (delta*0.8)
    logging.info("acquired frequency bins")
    return freq

def BinFreqArray(freq_array, power, bins, axion_freq):
    binned_freq = {}
    binned_pow = {}
    for i in range (0,len(bins[0])):
        lower_index = np.searchsorted(freq_array, bins[0][i])
        upper_index = np.searchsorted(freq_array, bins[1][i])
        binned_freq[axion_freq[i]] = freq_array[lower_index:upper_index]
        binned_pow[axion_freq[i]] = power[lower_index:upper_index]
        if i%100000 == 0:
            logging.info("Chunking "+str(i)+"/"+str(len(bins[0]))+" freq")
            if axion_freq[i] in binned_freq: logging.info("Chunk of length "+str(np.shape(binned_freq[axion_freq[i]])))
    return binned_freq, binned_pow

def PlotBinnedFreq(dict_freq, dict_pow, key_freq):
    fig = plt.figure(figsize=(8,4))
    if not(key_freq in dict_freq):
        logging.error("Frequency key not in binned dictionary")
    else:
        plt.xlabel(r'Frequency [Hz]')
        plt.ylabel(r'Amplitude $[mV^2/Hz]$')
        plt.title('Binned Spectra about f = '+str(np.rint(key_freq)))
        plt.plot(dict_freq[key_freq],dict_pow[key_freq],label='Binned Spectra, f = '+str(np.rint(key_freq)))
        plt.legend()
        plt.grid()
        plt.show()

def GetRescaledData(freq_window, data, template, ma):
    fmin = ma / 2. / np.pi * (1-.5*(vObs + v0)**2)
    fmax = ma / 2. / np.pi * (1+2*(vObs + v0)**2)
    if(fmin < np.amin(freq_window)): 
        fmin = np.amin(freq_window)
    if(fmax > np.amax(freq_window)): 
        fmax = np.amax(freq_window)
    
    locs = np.where((freq_window >= fmin)*(freq_window <= fmax))[0]
    
    freq_center = (np.amax(freq_window[locs]) + np.amin(freq_window[locs])) / 2
    freq_width = np.amax(freq_window[locs]) - np.amin(freq_window[locs])
    freqs_no_dim = 2*(freq_window[locs]-freq_center) / freq_width

    data_rescale = np.mean(data)
    data_no_dim = np.reshape(data, (1,len(data)))/data_rescale
    
    template_rescale = np.amax(template)
    template_no_dim = template / template_rescale

    scale_factors = [data_rescale, float(template_rescale)]
    return freqs_no_dim, data_no_dim, template_no_dim, scale_factors

def FitSignalModelHypothesis(freqs_no_dim, data_no_dim, template_no_dim, scale_factors, log = False):
    data = data_no_dim*scale_factors[0]

    amp_guess = 0
    amp_upper_bound = 2*(np.amax(data_no_dim) - np.amin(data_no_dim))
    amp_lower_bound = -2*(np.amax(data_no_dim) - np.amin(data_no_dim))

    mean_guess = np.mean(data_no_dim, axis = 1)
    mean_upper_bounds = 2*mean_guess
    mean_lower_bounds = np.zeros_like(mean_guess)

    slope_guess = np.zeros_like(mean_guess)
    slope_upper_bounds = .25*np.ones_like(slope_guess)
    slope_lower_bounds = -.25*np.ones_like(slope_guess)

    upper_bounds = np.append(amp_upper_bound, np.append(mean_upper_bounds, slope_upper_bounds))
    lower_bounds = np.append(amp_lower_bound, np.append(mean_lower_bounds, slope_lower_bounds))
    bounds = np.vstack((lower_bounds, upper_bounds)).T
    guess = np.append(0, np.append(mean_guess, slope_guess))

    loss = lambda x: analysis.NegLL(x, freqs_no_dim, data_no_dim, template_no_dim)
    grad = lambda x: analysis.NegLL_Jac(x, freqs_no_dim, data_no_dim, template_no_dim)

    m = Minuit(loss, guess, grad = grad)
    m.errordef = 1
    m.limits = bounds

    # Initially fix all parameters
    for i in range(len(guess)):
        m.fixed['x' + str(i)] = True
    
    for i in range(data_no_dim.shape[0]):
        m.fixed['x' + str(i+1)] = False
        m.fixed['x' + str(i + 1 + data_no_dim.shape[0])] = False
        m.migrad()
    
        m.fixed['x' + str(i+1)] = True
        m.fixed['x' + str(i + 1 + data_no_dim.shape[0])] = True
    
    null_TS = m.fval
    null_Fit = np.array(m.values)

    m = Minuit(loss, null_Fit, grad = grad)
    m.errordef = 1
    m.limits = bounds
    m.migrad()

    signal_TS = m.fval
    signal_Fit = m.values

    discovery_TS = null_TS - signal_TS
    best_fit = m.values['x0'] / scale_factors[1] * scale_factors[0]
    error = m.errors['x0'] / scale_factors[1] * scale_factors[0]
    if log == True:
        logging.info("Discovery TS:" + str(discovery_TS)+", best fit: "+ str(best_fit)+ ", error: " + str(error))
    
    return discovery_TS, best_fit, error, m

def GetTwoSigmaConfidence_OneMass(target_freq, freq_window, data, plot=False, log = False):
    ma = 2*np.pi*target_freq #mass of the axion we are searching for
    
    template = analysis.getAxionTemplate(ma, freq_window, v0, vObs)

    freqs_no_dim, data_no_dim, template_no_dim, scale_factors = GetRescaledData(freq_window, data, template, ma)
    
    discovery_TS, best_fit, error, m = FitSignalModelHypothesis(freqs_no_dim, data_no_dim, template_no_dim, scale_factors, log)
    try:
        m.minos('x0', cl = .9)
    except RuntimeError as e:
        logging.error("Couldn't find best fit for mass point "+str(target_freq))
        return [0,0,0,0]

    limit =  m.merrors['x0'].upper / scale_factors[1] * scale_factors[0]

    # One Sigma Upper Errorbar for Limit
    m.minos('x0', cl = stats.chi2.cdf((stats.chi2.ppf(.9, df = 1)**.5+1)**2, df = 1))
    one_sigma_upper =  m.merrors['x0'].upper / scale_factors[1] * scale_factors[0]

    # Two Sigma Upper Errorbar for Limit
    m.minos('x0', cl = stats.chi2.cdf((stats.chi2.ppf(.9, df = 1)**.5+2)**2, df = 1))
    two_sigma_upper =  m.merrors['x0'].upper / scale_factors[1] * scale_factors[0]

    # One Sigma Lower Errorbar for Limit
    m.minos('x0', cl = stats.chi2.cdf((stats.chi2.ppf(.9, df = 1)**.5-1)**2, df = 1))
    one_sigma_lower =  m.merrors['x0'].upper / scale_factors[1] * scale_factors[0]

    # Now we'll use Minuit to construct a profiled likelihood
    out = np.array(m.mnprofile('x0', size = 100, bound = 6, subtract_min = True))
    
    # Finally, use Asimov to get the power-constrained limit threshold
    pcl = (stats.chi2.ppf(.9, df = 1)**.5-1)*m.errors['x0']/ scale_factors[1] * scale_factors[0]
    limit = np.maximum(limit, pcl) 

    if plot == True:
        mpl.rcParams['figure.figsize'] = 8, 4
        plt.plot(out[0]/ scale_factors[1] * scale_factors[0], out[1], lw = 2, color = 'black')
        plt.axvline(best_fit, color = 'black', ls = ':', lw = 2, label = 'Best Fit')
        plt.axvline(limit, color = 'red', lw = 3, label = '95\% Upper Limit')
        plt.axvline(pcl, color = 'navy', ls = '--', lw = 2, label = 'PCL Threshold')
        plt.axvspan(one_sigma_lower, one_sigma_upper, color = 'limegreen', alpha = .75)
        plt.axvspan(one_sigma_upper, two_sigma_upper, color = 'gold', alpha = .75)

        #plt.ylim(0, 25)
        plt.ylabel('Test Statistic')
        plt.xlabel('Signal Amplitude')
        plt.legend(fontsize = 16, ncol =3, loc = 'upper left')
        plt.show()
    return [one_sigma_lower, one_sigma_upper, two_sigma_upper, limit]

def getBrazilBand(path, files, out, level = 'coarse'):
    if type(files) == list:
        len_f = 200 * len(files) #this is a problem!
        freq, psd = GetAveragePSDfromTS(path,
                                    files,
                                    1,
                                    N=100000000,
                                    plot = False)
        np.save(str(out)+'.npy',[freq,psd])   
    elif files.endswith(".npy"):
        freq, psd = np.load(path+files)
    else:
        logging.error("Incorrect file format")
    coarse = False
    n = 1
    if (level == 'coarse') or (level == None):
        coarse = True
    elif level == 'fine':
        downsample = 1
        n = 10000
        logging.info("fine scan analysis")
    else:
        logging.error("Invalid level. Choose 'coarse' or 'fine')")
    axion_freqs = GetSearchFreqArray(abra = coarse)
    bins = GetFreqBins(axion_freqs)
    logging.info("Produced search array and bins.")
    binned_frequencies, binned_powers = BinFreqArray(freq, psd, bins, axion_freqs)
    limits_dict = {}
    log = False
    count = 0
    for axion_candidate in axion_freqs:
        if not(axion_candidate in binned_frequencies):
            logging.error("Frequency key not in binned dictionary")
        else:
            if count%100 == 0:
                logging.info(axion_candidate)
                log = True
            else:
                log = False
            limits_calc = GetTwoSigmaConfidence_OneMass(axion_candidate, 
                                                        binned_frequencies[axion_candidate], 
                                                        binned_powers[axion_candidate],
                                                        plot=False, log = log)
            limits_dict[axion_candidate] = limits_calc
        count+=1

    one_sigma_lower = np.zeros(shape=(len(limits_dict.keys())))
    limit = np.zeros(shape=(len(limits_dict.keys())))
    one_sigma_upper = np.zeros(shape=(len(limits_dict.keys())))
    two_sigma_upper= np.zeros(shape=(len(limits_dict.keys())))
    freq_save = np.array(list(limits_dict.keys()))
    mass = freq_save*h
    
    for i in range(0,len(freq_save)):
        one_sigma_lower[i] = (conv*limits_dict[list(limits_dict.keys())[i]][0]*freq_save[i])**.5
        limit[i] = (conv*limits_dict[list(limits_dict.keys())[i]][3]*freq_save[i])**.5
        one_sigma_upper[i] = (conv*limits_dict[list(limits_dict.keys())[i]][1]*freq_save[i])**.5
        two_sigma_upper[i] = (conv*limits_dict[list(limits_dict.keys())[i]][2]*freq_save[i])**.5
    one_sigma_lower_interp=interp1d(mass[::n], one_sigma_lower[::n])
    one_sigma_upper_interp=interp1d(mass[::n], one_sigma_upper[::n])
    two_sigma_upper_interp=interp1d(mass[::n], two_sigma_upper[::n])
    limits_interp=interp1d(mass[::n], limit[::n])

    
    logging.info("Produced Mass Array and Limits")
    
    #save the data
    assert len(mass) == len(one_sigma_lower) == len(one_sigma_upper) == len(two_sigma_upper) == len(limit)
    combo_array = np.column_stack((mass, one_sigma_lower, one_sigma_upper, two_sigma_upper, limit))
    combo_array = combo_array.reshape((-1,5))
    output_file = str(out)+".csv"
    np.savetxt(output_file, combo_array, delimiter=",")

    fig = plt.figure(figsize=(6,3))

    mass_plot = mass[:-n:int(len(mass)/1000)]
    plt.fill_between(mass_plot, one_sigma_lower_interp(mass_plot), one_sigma_upper_interp(mass_plot), color = 'limegreen', alpha = .75)
    plt.fill_between(mass_plot, one_sigma_upper_interp(mass_plot), two_sigma_upper_interp(mass_plot), color = 'gold', alpha = .75)
    plt.plot(mass_plot,limits_interp(mass_plot), color = 'black', alpha = 1, lw=0.5, label = '95\% Upper Limit')

    plt.yscale('log')
    plt.ylabel(r'$g_{a\gamma\gamma}$ (1/GeV)')
    plt.xscale('log')
    plt.xlabel('Axion Mass (eV)')
    
    plt.savefig(str(out)+".png",dpi=600,bbox_inches='tight')

def read_file_names(file_path):
    with open(file_path, 'r') as file:
        file_names = file.readlines()
        
        # Remove whitespace characters like '\n' at the end of each line
        file_names = [file_name.strip() for file_name in file_names]
    return file_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full axion analysis on data.")
    parser.add_argument("path", type=str, help="pathway to input files")
    parser.add_argument("files", type=str, help=".txt file containing all of the .h5 file names. Or, if psd averaging has been done, the .npy file containing [freq, pwr].")
    parser.add_argument("out", type=str, help="file name for output brazil band plot. Plot will be saved as '[out].png'. Data will be saved at '[out].csv'")
    parser.add_argument("--level", type=str, help="either 'coarse' or 'fine' for coarse or full axion mass points")
    parser.add_argument("--v", action='store_true', help="verbose option for logger")
    args = parser.parse_args()
    
    if(os.path.splitext(args.files)[1] == ".txt"):
        logging.info("Will compute psd")
        file_names = read_file_names(args.files)
        logging.info('Running analysis over ' + str(file_names))
    elif(os.path.splitext(args.files)[1] == ".npy"):
        logging.info("Will NOT compute psd")
        file_names = args.files
    else:
        logging.error("Wrong input file type. --h for help.")

    if args.v:
        logging.getLogger().setLevel(logging.INFO)
        
    getBrazilBand(args.path, file_names, args.out, level = args.level)
