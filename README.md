# wavecast: A forecasting model based on WaveNet architecture

This work has been done as part of Master's thesis titled "WaveNet Architectures for Time Series Forecasting".

**Author**: Naveen Kaushik

**Supervisor**: [Dr. Christoph Bergemeir](https://www.cbergmeir.com/)

**Published date**: July 15, 2020

# Installation and Usage

### Software Requirements:
python>=3.6

tensorflow==1.13

keras==2.2.4


### Usage:

`python wavecast.py --dataset_name "dummy_ts_data" --train_file "train.csv" --test_file "test.csv" --training_steps 72 --forecast_horizon 12 --output_file "forecast.csv" --frac 1.0 --result_file "results.csv"`

The parameters used are explained as follows:

- dataset_name - a unique string to identify the dataset
- train_file - the training data file for training of the model
- test_file - the test data file for evaluating the model
- training_steps - number of timesteps to be considered for training window
- forecast_horizon - the forecasting horizon for the dataset
- output_file - file to write the forecasted values
- frac - fraction of input data to be used
- result_file - file to write the results based on evaluation metric and hyperparameter values

Most of the datasets used in the experiments are taken from [here](https://zenodo.org/communities/forecasting?page=1&size=20).
