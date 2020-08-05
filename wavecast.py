
"""
Code written as part of Master's thesis 
titled "WaveNet Architectures for Time Series Forecasting".

Author: Naveen Kaushik
Published date: July 15, 2020

Please refer to the following link for detailed experiments and results:
https://github.com/naveenkaushik2504/wavecast/blob/master/Masters_Thesis.pdf


Sources/References for WaveNet implementations:
1. https://github.com/kristpapadopoulos/seriesnet/blob/master/seriesnet.py
2. https://gist.github.com/jkleint/1d878d0401b28b281eb75016ed29f2ee
3. https://github.com/llSourcell/Music_Generation/blob/master/simple-generative-model-regressor.py


"""

from keras.models import Model, load_model
from keras.layers import Input, Add, Activation, Conv1D, Dense, Dropout, Lambda, Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K

from keras.initializers import TruncatedNormal
from keras.regularizers import l2
from keras.layers.advanced_activations import LeakyReLU, ELU

# hyperopt 
from hyperopt import hp
from hyperopt import fmin, tpe, space_eval

import operator
import collections
import numpy as np
import json
from numpy.random import seed
import random
import os
import glob
import argparse
import csv
from tqdm import tqdm
from datetime import datetime


seed_value = 16

tf.set_random_seed(seed_value)
random.seed(seed_value)
seed(seed_value)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

def read_data(file_name, frac):
    """Read data from a given file

    Args:
        file_name (str): name of the file
        frac (float): fraction of total data to be considered
    
    Returns:
        data_ts (list): list of time series
    """

    print("reading data from:", file_name)
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        data_ts = list(reader)
    
    # TODO: code for checking Nans
    data_ts = [list(map(float, x)) for x in data_ts]

    # Sample the data
    if frac != 1.0:
        data_ts = random.sample(data_ts, int(len(data_ts)*frac))

    return data_ts


series_total_len = 1000
pred_steps = 48
val_steps = 48


def create_moving_window(use_ts):
    """Convert data to a moving window format

    Args:
        use_ts (list): list of time series
    
    Returns:
        window_ts (list): list of time series in a moving window format
    """

    window_ts = []
    skipped = 0
    for item in use_ts:
        ts_step = len(item) - series_total_len
        if ts_step < 0:
            skipped += 1
            continue
        for i in range(ts_step):
            temp_ts = item[i:series_total_len+i]
            window_ts.append(temp_ts)
    print("Number of series skipped: ", skipped)
    return window_ts
    
    
def log_transform(use_ts):
    """Convert data to a moving window format

    Args:
        use_ts (list): list of time series
    
    Returns:
        window_ts (list): list of logged time series
    """

    log_data = [np.log(series) for series in use_ts]
    return log_data


def create_training_data(use_ts, train_steps, pred_steps):
    """Creates the data for training of the model

    Args:
        use_ts (list): list of time series
        train_steps (int): time steps for training
        pred_steps (int): time steps for prediction
    
    Returns:
        x_train (np.array): x for training
        y_train (np.array): y for training
        x_val (np.array): x for validation
        y_val (np.array): y for validation
    """

    # Considering 70% of the data due to the limitation of computational resources
    train_points = int(len(use_ts)*0.7) 
    train_data = use_ts[0:train_points]
    test_data = use_ts[train_points: ]
    x_train = np.array([t[:train_steps] for t in train_data])
    y_train = np.array([t[train_steps:train_steps + pred_steps] for t in train_data])
    x_val = np.array([t[:train_steps] for t in test_data])
    y_val = np.array([t[train_steps:train_steps + pred_steps] for t in test_data])

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
    
    return x_train, y_train, x_val, y_val

def create_test_data(use_ts, train_steps, pred_steps):
    """Creates the data for testing of the model

    Args:
        use_ts (list): list of time series
        train_steps (int): time steps for training
        pred_steps (int): time steps for prediction
    
    Returns:
        x_test (np.array): x for testing
        y_test (np.array): y for testing
    """

    x_test = np.array([t[:train_steps] for t in use_ts])
    y_test = np.array([t[train_steps:train_steps + pred_steps] for t in use_ts])

    return x_test, y_test


def DC_CNN_Block(nb_filter, filter_length, dilation, l2_layer_reg):
    """Creates a block of dilated 1-D network

    Args:
        nb_filter (int): number of filters
        filter_length (int): length of convolutional filter
        dilation (int): dilation factor
        l2_layer_reg (float): time steps for training
    
    Returns:
        one block of WaveNet model
    """

    def f(input_):
        
        residual =    input_
        
        layer_out =   Conv1D(filters=nb_filter, kernel_size=filter_length, 
                      dilation_rate=dilation, activation='linear', padding='causal', 
                      kernel_regularizer=l2(l2_layer_reg))(input_)
                    
        layer_out =   Activation('relu')(layer_out)
        
        skip_out =    Conv1D(1,1, activation='linear', 
                      kernel_regularizer=l2(l2_layer_reg))(layer_out)
        
        network_in =  Conv1D(1,1, activation='linear', 
                      kernel_regularizer=l2(l2_layer_reg))(layer_out)
                      
        network_out = Add()([residual, network_in])
        
        return network_out, skip_out
    
    return f

def DC_CNN_Model(train_steps, forecast_horizon, configs):
    """Creates the WaveNet model using individual blocks

    Args:
        train_steps (int): number of steps for training
        forecast_horizon (int): number of steps for forecasting
        configs (dict): dictionary containing the hyperparameters
    
    Returns:
        model (Model): model object based on the network
    """

    lr = configs['learning_rate']
    dropout_rate = configs['dropout_rate']
    num_filters = int(configs['num_filters'])
    dropout_training = False #bool(configs['dropout_training'])
    filter_length = int(configs['filter_length'])
    l2_layer_reg = configs['l2_layer_reg']

    input = Input(shape=(train_steps,1))
    
    # The number of dilated blocks would vary based on the input window size
    l1a, l1b = DC_CNN_Block(num_filters,filter_length,1,l2_layer_reg)(input)    
    l2a, l2b = DC_CNN_Block(num_filters,filter_length,2,l2_layer_reg)(l1a) 
    l3a, l3b = DC_CNN_Block(num_filters,filter_length,4,l2_layer_reg)(l2a)
    l4a, l4b = DC_CNN_Block(num_filters,filter_length,8,l2_layer_reg)(l3a)
    l5a, l5b = DC_CNN_Block(num_filters,filter_length,16,l2_layer_reg)(l4a)
    l6a, l6b = DC_CNN_Block(num_filters,filter_length,32,l2_layer_reg)(l5a)
    l6b = Dropout(dropout_rate)(l6b, training = dropout_training) 
    l7a, l7b = DC_CNN_Block(num_filters,filter_length,64,l2_layer_reg)(l6a)
    l7b = Dropout(dropout_rate)(l7b, training = dropout_training) 

    l8 =   Add()([l1b, l2b, l3b, l4b, l5b, l6b, l7b])
    
    l9 =   Activation('relu')(l8)
           
    l21 =  Conv1D(1,1, activation='linear', 
           kernel_regularizer=l2(0.001))(l9)
    l31 = Flatten()(l21)
    l41 = Dense(forecast_horizon, activation='linear')(l31)

    model = Model(input=input, output=l41)
    
    adam = Adam(lr=lr)

    model.compile(loss='mae', optimizer=adam, metrics=['mae'])
    
    return model


def smape(actual, forecast):
    """Calculate the SMAPE(Symmetric Mean Absolute Error)

    Args:
        actual (np.array): actual values
        forecast (np.array): forecasted values
    
    Returns:
        float : smape error
    """

    return 100/len(actual) * np.sum(2 * np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast)))
    
def get_error(y_true, y_pred):
    """Calculate the mean SMAPE error

    Args:
        y_true (np.array): actual values
        y_pred (np.array): forecasted values
    
    Returns:
        float : mean smape error
    """

    error_list = []
    for i in range(len(y_pred)):
        error_list.append(smape(y_true[i], y_pred[i]))
    
    return np.mean(error_list)
    
def write_forecasts(y_pred, out_file):
    """Write the forecasted value in a file

    Args:
        y_pred (np.array): forecasted values
        out_file (str): output file name
    
    Returns:
        None
    """

    with open(out_file, 'w', newline='') as myfile:
        writer = csv.writer(myfile)
        writer.writerows(y_pred)
        
# Training the time series
def objective(configs):
    """Train model for each iteration of hyperopt

    Args:
        configs (dict): dictionary containing the hyperparameters
    
    Returns:
        error (float): error for the iteration
    """

    error, _ = train_model(configs)
    return error
    
# Individual model training
def train_model(configs):
    """Train model 

    Args:
        configs (dict): dictionary containing the hyperparameters
    
    Returns:
        error (float): error for the iteration
        model (Model): trained model object
    """

    print("Training model")
    print(configs)
    model = DC_CNN_Model(train_steps, forecast_horizon, configs)
    model.summary()
    
    batch_size = 25 #int(configs['batch_size'])
    epochs = 4
    print("Batch size:", batch_size)
    print(x_train.shape)
    print(y_train.shape)
    model.fit(x_train, y_train, epochs=epochs,\
        batch_size=batch_size, validation_split=0.2)

    print("After fit...")

    y_val_pred = np.exp(model.predict(x_val))
    print(y_val_pred[0], y_val_pred[1])
    print(y_val[0], y_val[1])
    error = get_error(np.exp(y_val), y_val_pred)
    print("Error is ", str(error))

    return error, model
    

def optimize():
    """Hyperparameter optimization

    Args:
        None
    
    Returns:
        best (dict): tuned hyperparameters
    """

    space = {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.01)),
        'dropout_rate': hp.uniform('dropout_rate', 0.2, 0.8),
        # 'dropout_training': hp.choice('dropout_training', [True, False]),
        'num_filters': hp.quniform('num_filters', 20, 80, 5),
        'filter_length': hp.quniform('filter_length', 1, 10, 1),
        'l2_layer_reg': hp.uniform('l2_layer_reg', 0.0, 1.0),
        # 'batch_size': hp.quniform('batch_size', 20, 80, 5),
        }

    best = fmin(objective, space, algo=tpe.suggest, max_evals=4)

    print(best)
    return best
	

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Parameters for model training:")
    parser.add_argument('--dataset_name', required=True, help='Unique string for the name of the dataset')
    parser.add_argument('--train_file', required=True, help='Path for training file')
    parser.add_argument('--test_file', required=True, help='Path for test dataset')
    parser.add_argument('--training_steps', required=True, help='Number of time steps for training')
    parser.add_argument('--forecast_horizon', required=True, help='Number of time steps for prediction')
    parser.add_argument('--output_file', required=True, help='Output file to write forecasts')
    parser.add_argument('--frac', required=False, help='Fraction of data to consider', default=0.2)
    parser.add_argument('--result_file', required=True, help='Result file to write errors')
    
    args = parser.parse_args()
    print(args)
    
    result_file = args.result_file
    
    train_steps = int(args.training_steps) 
    forecast_horizon = int(args.forecast_horizon)
    series_total_len = train_steps + forecast_horizon

    seed_value = 16

    tf.set_random_seed(seed_value)
    random.seed(seed_value)
    seed(seed_value)

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    
    #read the training series
    frac = float(args.frac)
    use_ts = read_data(args.train_file, frac)
    
    #create moving window data
    use_ts = create_moving_window(use_ts)
    
    #TODO: Mean scaling
    
    #Log transform the data
    use_ts = log_transform(use_ts)
    print(use_ts[0])
    
    #create training data
    x_train, y_train, x_val, y_val = create_training_data(use_ts, train_steps,forecast_horizon)
    print(x_train.shape)
    print(y_train.shape)
    print(x_val.shape)
    print(y_val.shape)

    tuned_params = optimize()
    
    # Run the training for different seeds
    error_list = []
    for i in range(2):
        seed_value = random.randint(1, 10)
        print("Running for seed ", int(seed_value))

        tf.set_random_seed(seed_value)
        random.seed(seed_value)
        seed(seed_value)

        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)
        error, model = train_model(tuned_params)
        
        #Read and build test data
        train_data = read_data(args.train_file, 1.0)
        test_data = read_data(args.test_file, 1.0)
        
        #Create test data
        test_ts = []
        skipped = 0
        for ts_train, ts_test in zip(train_data, test_data):
            ts = np.concatenate([ts_train[-train_steps:], ts_test])
            if len(ts) < train_steps + forecast_horizon:
                skipped += 1
                temp_ts = [1.0] * (train_steps + forecast_horizon - len(ts))
                ts = temp_ts + list(ts)
            test_ts.append(ts)
        print("modified time series due to length", skipped)
        print(test_ts[0])
        
        #Log transform the data
        test_ts = log_transform(test_ts)
        print(test_ts[0])
        
        #Create test data
        x_test, y_test = create_test_data(test_ts, train_steps, forecast_horizon)
        print(x_test.shape)
        print(y_test.shape)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
        
        #Predict test data
        n_samples = 10
        y_pred = []
        for i in tqdm(range(n_samples)):
            out_sample = np.exp(model.predict(x_test))
            y_pred.append(out_sample)
            
        y_pred = np.median(y_pred, axis = 0)
        
        #Write the forecast to output file
        write_forecasts(y_pred, args.output_file)
        
        #calculate error
        error = get_error(test_data, y_pred)
        error_list.append(error)
        
        print("error list so far is: ", str(error_list))
    
    final_error = np.mean(error_list)
    #Write results
    with open(result_file, 'a') as res_file:
        res_file.write(str(args.dataset_name) + ',' + str(datetime.now()) + ',' \
            + str(train_steps) + ',' + str(tuned_params) + ',' + str(final_error) + '\n' )
    
