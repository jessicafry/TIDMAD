# -*- coding: utf-8 -*-
"""
Created on Mon May 10 2024
@author: Aobo Li
This script downloads the TIDMAD data generated by the ABRACADABRA dark matter detectors. 
The downloading action is executed with wget command in linux.
"""

import argparse
import os
import urllib
import subprocess


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

def download_type(folder, file_prefix, num_files, args):
	"""Download data with the given types"""
	# Generate cache domain name;
	if args.cache == "SoCal":
		cache_domain = "stashcache.t2.ucsd.edu:8443"
	elif args.cache == "NorCal":
		cache_domain = "osg-sunnyvale-stashcache.t2.ucsd.edu:8443"
	elif args.cache == "NY":
		cache_domain = "osg-new-york-stashcache.nrp.internet2.edu:8443"
	elif args.cache == "Director":
		cache_domain = "osdf-director.osg-htc.org"
	else:
		raise argparse.ArgumentTypeError(f"Invalid argument {args.cache} for cache domain selection! Please choose from [NY/SoCal/NorCal]")


	# Download data files with the given argmument
	start_i = 20 if (args.weak) and ("science" not in file_prefix) else 0
	for i in range(start_i, start_i+num_files):
		if i < 10:
			fname = file_prefix+"000%d.h5"%(i)
		elif i<100:
			fname = file_prefix+"00%d.h5"%(i)
		else:
			fname = file_prefix+"0%d.h5"%(i)
		url = f"https://{cache_domain}/ucsd/physics/ABRACADABRA/{folder}/{fname}"
		# path = args.output_dir
		if args.skip_downloaded and os.path.exists(os.path.join(args.output_dir,fname)):
			continue
		if args.print:
			print(f"wget {url}")
		else:
			try:
				subprocess.run(["wget", url, "-O", os.path.join(args.output_dir,fname)])
			except urllib.error.HTTPError as err:
				print(f"File {fname} not downloaded! Please consider trying with another cache using the -c flag.")
	return 0

def main():
	parser = argparse.ArgumentParser(description="Download the DarkSense dataset generated by the ABRACADABRA experiment. Each data file is a 2Gb hdf5 format.")

	# Output directory with default as current directory
	parser.add_argument('--output_dir', '-o', type=str, default=os.getcwd(), 
	                    help='Directory where the files will be saved (default: current directory).')
	parser.add_argument('--cache', '-c', type=str, default="Director", 
	                    help='Which OSDF cache should be used to download data. Options are [NY/NorCal/SoCal/Director(default)]: New York [NY], Sunnyvale (NorCal), San Diego (SoCal), Director (automatically allocate cache based on user\'s location. WARNING: Director cache is sometimes unstable, we recommend switching to a different cache if it fails downloading.')
	# Integer flags for numbers of files to download with range check for train files
	parser.add_argument('--train_files', '-t', type=check_range(0, 20), default=20, 
	                    help='Number of training files to download, must be an  integer between 0 and 20, default 20.')
	parser.add_argument('--validation_files', '-v', type=check_range(0, 20), default=20, 
	                    help='Number of validation files to download, must be an  integer between 0 and 20, default 20.')
	parser.add_argument('--science_files', '-s', type=check_range(0, 208), default=208, 
	                    help='Number of science files to download, must be an  integer between 0 and 208, default 208.')
	parser.add_argument('-f', '--force', action='store_true', help='Directly proceed to download without showing the file size and confirm.')
	parser.add_argument('-sk', '--skip_downloaded', action='store_true', help='Skip the file that has already been downloaded at --output_dir')
	parser.add_argument('-w', '--weak', action='store_true', help='Download the "weak signal" version of training and validation file. In this version, the injected signal is 1/5 of the normal version. This is a more challenging denoising task.')
	parser.add_argument('-p', '--print', action='store_true', help='Print out all downloading scripts instead of actually downloading the file.')

	args = parser.parse_args()

	print(f"Output directory: {args.output_dir}")
	print(f"Training files to download: {args.train_files}")
	print(f"Validation files to download: {args.validation_files}")
	print(f"Physics files to download: {args.science_files}")

	print(f"Estimated total file size to download is : {args.train_files*2.06+args.validation_files*2.06+args.science_files*2.8} Gb")
	if (not args.force) and (not args.print) :
		response = input("Do you want to proceed? [y/n] ").strip().lower()
		if response in ['no', 'n']:
	            return 0

	if not os.path.exists(args.output_dir):
		raise argparse.ArgumentTypeError(f"Download-to directory {args.output_dir} does not exist.")

	download_type("ABRA_aires_training_data", "abra_training_", args.train_files, args)
	download_type("ABRA_aires_validation_data", "abra_validation_", args.validation_files, args)
	download_type("ABRA_aires_science_data", "abra_science_", args.science_files,  args)

if __name__ == "__main__":
    main()