# VitalML: Predicting Patient Decompensation from Continuous Physiologic Monitoring in the Emergency Department

## Getting started 

After cloning the repo, to install dependencies from the `environment.yml` file, run:

`conda env create -f environment.yml`

## Running VitalML

Update all data source paths in `path_configs.json`

### Training transformer models
See `transformer/training_example.sh` for an example of a slurm script that runs the `transformer/training.py` program. Arguments are as follows:

`python training.py <waveform embedding size> <task> <waveform lead> <timepoint>`

Waveform embedding size is the embedding size in the layer directly prior to the output. Task is the decompensation prediction task, so one of `HR`, `SpO2`, `MAP` or `MEWS`. Waveform lead specifies the usage of `ECG`, `Pleth` or `All`. Lastly, timepoint indicates the length of the evaluation period for predicting decompensation and is one of `60min`, `90min` or `120min`. 

### Training and tuning LGBM models 

To train and tune models across all configurations of input data, run:

`python lgbm/prediction.py`

The above script will also evaluate the best model on the test set after tuning and selection of the best random seed. However, to bootstrap the best model's performance across the test set, run the following:

`python lgbm/bootstrap_results.py`

### Model interpretation and analysis 

To run secondary analyses on the your models of interest (in our case the baseline and best-performing models shown in `best_model_configs.json`, run:

`python analysis/run_analysis <analysis_mode>`

The following analysis modes are available:

- `shap`: Identifies highest contributing features to model prediction
- `mews`: Performs subgroup analysis on model decompensation predictions based on ground truth MEWS score labels
- `calibration_curve`: Graphs calibration plots
- `characteristic`: Computes model characteristics such as sensitvity, specifity, PPV and NPV on the test dataset
- `confusion`: Calculates confusion matrix of model predictions on the test dataset

## License
The source code for the site is licensed under the MIT license, which you can find in the LICENSE file.
