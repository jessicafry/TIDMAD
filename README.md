# TIDMAD
Code repository for TIDMAD: Time series Dataset for Discovering Dark Matter with AI Denoising.<br>__TL;DR:__  TIDMAD is the first dataset and benchmark from a dark matter physics experiment, providing ultra-long time series data and comprehensive tools that enable machine learning models to directly advance the fundamental physics search for dark matter.

## Download Data
The TIDMAD dataset can be downloaded using the download_data.py script provided in this GitHub repository. This script runs without any external dependencies. Data downloading is achieved by generating a series of wget commands and executing them in a Linux environment. download_data.py has the following arguments:
* `--output_dir -o`: Destination directory where the file will be saved, default: current working directory.
* `--cache -c` Which OSDF cache location should be used to download data. Options includes [NY/NorCal/SoCal/Director(default)]:
  * NY: New York
  * NorCal: Sunnyvale
  * SoCal: San Diego
  * Director: automatically allocatecache based on user's location. WARNING: Director cache is sometimes unstable, we recommend switching to a different cache if the downloading fails.
* `--train_files -t`: Number of training files to download, must be an integer between 0 and 20, default 20.
* `--validation_files -v`: Number of validation files to download, must be an integer between 0 and 20, default 20.
* `--science_files -s`: Number of science files to download, must be an integer between 0 and 208, default 208.
* `-f, --force`: Directly proceed to download without showing the file size and asking the confirmation question.
* `-sk, --skip_downloaded`: Skip the file that has already exist at --output_dir
* `-w, --weak`: Download the weak signal version of training and validation file. In this version, the injected signal is 1/5 amplitude of the normal version. This is a more challenging denoising task. Please note that the normal version has file range 0000-0019, while the weak version has file range of 0020-0039
* `-p, --print`           Print out all wget commands instead of actually executing the downloading command.

Alternatively, you can use the filelist.dat file, which contains the wget commands to download the entire dataset (except the weak signal version).

## Dataset Composition:
The dataset includes 248 files (288 if the weak signal version is included), all in HDF5 format. The dataset is partitioned into three subsets:
1. Training Dataset: `abra_training_00{##}.h5` where ## varies from 00 to 20
   * Each training `.h5` file has the following format:<br><img width="274" alt="Screenshot 2024-06-02 at 6 49 50 PM" src="https://github.com/aobol/TIDMAD/assets/25975621/0f99b6e6-2f7c-4566-91e8-cc29985f32c2">
2. Training Dataset: `abra_validation_00{##}.h5` where ## varies from 00 to 20
   * Each validation `.h5` file has the following format:<br><img width="279" alt="Screenshot 2024-06-02 at 6 51 55 PM" src="https://github.com/aobol/TIDMAD/assets/25975621/fe466977-fe9c-46c2-8186-11986ed7a3c0">
2. Science Dataset: `abra_science_0{###}.h5` where ### varies from 000 to 207
   * Each science `.h5` file has the following format:<br><img width="273" alt="Screenshot 2024-06-02 at 6 52 52 PM" src="https://github.com/aobol/TIDMAD/assets/25975621/17f8ae17-6942-4840-a092-a6d268fc2d83">
   * For science files, there are no injected fake signal therefore only 1 channel is present

**Caveat:** due to a hardware condition, the size of channel0001 and channel0002 time series in a few training and validation files are not identical. This does not affect the sample-to-sample correspondence between two channels except in the last few time samples. We recommend only using the first `2000000000` samples in both channel for all files (i.e. `ch01_time_series = ch01_time_series[:2000000000]`).

## Model Training and Benchmarking:
<br>![dataflow](https://github.com/aobol/TIDMAD/assets/25975621/f9962719-8a65-4321-8c97-fea6d67f24b1)
<br>
TIDMAD users could follow the procedure below to reprocued the result in our paper:
1. Run `python download.py` script to download all datasets
2. Set up the required environment using `python setup.py install`
3. Train 3 deep learning models by running `python train.py -d [directory] -m [model]`
   * `[directory]` is where all the training files are downloaded to in step 1
   * `[model]` is the deep learning model to train, user should choose from `[fcnet/punet/transformer]`.
   * Note: for each deep learning model, 4 files will be produced due to Frequency Splitting discussed in the paper. (i.e. for `-m fcnet` there will be 4 files including `FCNet_0_4.pth`, `FCNet_4_10.pth`, `FCNet_10_15.pth`, `FCNet_15_20.pth`
   * Alternatively, users could download our trained model at Google Drive [Link](https://drive.google.com/drive/folders/16ORX1b2zo1_lOYYAcRBgddBuYImj0Bxs?usp=sharing)
4. Run `python inference.py -d [directory] -m [model]` to produced denoised time series file in `.h5` format
   * `[directory]` is where all the validation files are downloaded to in step 1.
   * For each validation file `abra_validation_00{##}.h5`, a denoised validation file `abra_validation_denoised_[model]_00{##}.h5` will be generated. Please note that the denoised validation file will also be saved at `[directory]`.
   * `[model]` is the denoising algorithm to run inference over, user should choose from `[mavg/savgol/fcnet/punet/transformer]`. If user choose one of `[fcnet/punet/transformer]`, the trained model file in `.pth` format must present. These file can be generated following step 3 or be downloaded directly.
5. Run `python denoising_score.py -d [directory] -m [model]`
6. Run `python process_science_data.py -d [directory] -m [model]` to generate the denoised time series over the science file (Hope please add these in)
7. Generate Dark Matter Limit plot  (Hope/Jessica/Mo please add details)


## Paper Abstract
Dark matter makes up approximately 85\% of total matter in our universe, yet it has never been directly observed in any laboratory on Earth as of today. The origin of dark matter is one of the most important questions in contemporary physics, and a convincing detection of dark matter would be a Nobel-Prize-level breakthrough in fundamental science. The ABRACADABRA experiment was meticulously designed to search for dark matter. Although it has not yet made a discovery, ABRACADABRA has produced several dark matter search result widely endorsed by the physics community. The experiment generates ultra-long time-series data at a rate of 10 million samples per second, where the dark matter signal, if exist, would manifest  itself as a sinusoidal oscillation mode within the ultra-long time series. In this paper, we present a comprehensive data release from the ABRACADABRA experiment including three key components: a ultra-long time series dataset divided into training, validation, and dark matter search subsets; a carefully-designed denoising score for direct model benchmarking; and a complete analysis framework which yield a community-standard dark matter search result suitable for publication in a physics journal. Our data release enables core AI algorithms to directly produce physics results thereby advancing fundamental science.

