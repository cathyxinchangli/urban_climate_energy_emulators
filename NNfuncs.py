import xarray as xr
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from datetime import datetime
from datetime import date

# Dictionaries that relate variable/scenario pair with the case name
case_var_dict = {
    'URBAN_AC_kJ': 'A',
    'URBAN_HEAT_kJ': 'H',
    'TSA_U': 'T'
}

case_num_dict = {
    'BSSP370cmip6': '1',
    'BSSP245cmip6': '2',
    'BSSP585cmip6': '3'
}

# Dictionary for variables & their names used for saving models etc
met_var_dict = {
    'URBAN_AC_kJ': 'AC',
    'URBAN_HEAT_kJ': 'HEAT',
    'TSA_U': 'TSA'
}

# Dictionary for the attributes of output file
attrs_dict = {
    'AC': {'long_name': 'urban air conditioning flux', 
           'units': 'kJ/m^2',
           'cell_methods': 'time: mean (NN predicted)'},
    'HEAT': {'long_name': 'urban heating flux',
             'units': 'kJ/m^2',
             'cell_methods': 'time: mean (NN predicted)'}, 
    'TSA': {'long_name': '2m air temperature (natveg, crop, TBD,HD,MD urban)',
            'units': 'K',
            'cell_methods': 'time: mean (NN predicted)'}
}

# Dictionary of the hyperparameters for each variable
hp_dict = {
    'AC': {'batch_size': 8,
           'epochs': 1200,
           'l1': 0.002,
           'l2': 1e-6,
           'learning_rate': 0.005,
           'output_activation': None},
    'HEAT': {'batch_size': 16,
             'epochs': 1000,
             'l1': 5e-6,
             'l2': 1e-6,
             'learning_rate': 0.005,
             'output_activation': None},
    'TSA': {'batch_size': 32,
            'epochs': 1200,
            'l1': 1e-6,
            'l2': 1e-6,
            'learning_rate': 0.0003,
            'output_activation': None}
}

# Functions for preprocessing and NN construction
def standardize(array, mean=None, std=None):
    '''
    Standardize TSA and/or other numerical features
    '''
    if mean is None:
        mean = array.mean(axis=0)
        std = array.std(axis=0)
    standardized = (array-mean)/std
    return standardized, mean, std

def destandardize(standardized, mean, std):
    '''
    Reverse standardized vars to original space, given training data's mean and std
    '''
    array = standardized*std+mean
    return array

def prepare_input(df, targets, features, is_train=True, to_standardize=True, mean_x_num = None, std_x_num = None):
    '''
    1. One-hot encode 'month';
    2. IF TRAIN: Shuffle samples;
    3. Standardize numerical features
    '''
    #1. & 2.
    if is_train:
        input_df = pd.get_dummies(df, columns=['month']).sample(frac=1)
    else:
        input_df = pd.get_dummies(df, columns=['month'])
    # Get targets data
    y = input_df[targets].values

    # Get feature data
    if to_standardize:
        x_num = input_df[features[:-1]].values
        x_cat = input_df[[i for i in input_df.columns if 'month' in i]].values
        x_num, mean_x_num, std_x_num = standardize(x_num, mean=mean_x_num, std=std_x_num)
        x = np.hstack((x_num, x_cat))
        return x, mean_x_num, std_x_num, y
    else:
        x = input_df[features[:-1]+[i for i in input_df.columns if 'month' in i]].values
        return x, y

def create_model(learning_rate, l1, l2, output_activation=None):
    model = tf.keras.Sequential() # a series of layers

    # Input layer
    model.add(tf.keras.layers.InputLayer(input_shape=(18,)))

    # Hidden layers - add non-linearity: activation function 
    model.add(tf.keras.layers.Dense(units=8, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l1_l2(l1=l1, l2=l2))) # Choose units as power of 2 
    #model.add(tf.keras.layers.Dropout(dropout_frac))
    model.add(tf.keras.layers.Dense(units=4, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l1_l2(l1=l1, l2=l2)))
    #model.add(tf.keras.layers.Dropout(dropout_frac))

    # Output layer
    model.add(tf.keras.layers.Dense(units=1, activation = output_activation))
    
    # Compiling model: what approach to do gradient descent
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate), #use shorthand "lr=" instead of "learning_rate=" will throw an error...
              loss='mse',
              metrics=[tf.keras.metrics.RootMeanSquaredError()]
                 )
    return model

def evaluate_performance(y, y_pred, verbose=True):
    '''
    Calculate the r^2 and RMSE of the predictions.
    '''
    rsq = r2_score(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)
    if verbose:
        print('r^2\tRMSE\t')
        print(f'{round(rsq, 3)}\t{round(rmse, 3)}\t')
    return rsq, rmse


# Functions for applying trained NN to other CMIP6 models
def prepare_input_apply(df, features, to_standardize=True, mean_x_num = None, std_x_num = None):
    '''
    1. One-hot encode 'month';
    2. IF TRAIN: Shuffle samples;
    3. Standardize numerical features
    '''
    #1. & 2.
    input_df = pd.get_dummies(df, columns=['month'])

    # Get feature data
    if to_standardize:
        x_num = input_df[features[:-1]].values
        x_cat = input_df[[i for i in input_df.columns if 'month' in i]].values
        x_num, mean_x_num, std_x_num = standardize(x_num, mean=mean_x_num, std=std_x_num)
        x = np.hstack((x_num, x_cat))
        return x, mean_x_num, std_x_num
    else:
        x = input_df[features[:-1]+[i for i in input_df.columns if 'month' in i]].values
        return x

def apply(var, ds, features, targets, train_dss, test_ds, pred_da, gridcells, savemodel_dir, testpred_path, start=0):
    '''
    For applying the saved NN models on another CMIP model, where no ground truths are available.
    -------
    ds: CMIP6 model outputs containing all features
    features: variables that NN uses to make predictions (note: CESM features are hard-coded in)
    targets: the target variable in the form of a sequence (list)
    pred_da: an empty dataset for storing the predicted values
    gridcells: urban gridcells to apply the NN models to, a pandas DataFrame
    '''
    start_time = datetime.now()
    print(f'Applying models started at {start_time}.')
    print('Applying models on ', end='')

    features_ = ['WIND', 'QBOT', 'TBOT', 'FLDS', 'FSDS', 'PRCP', 'month']
    for i, (lat, lon) in enumerate(gridcells.index):
        i_ = i+start

        lat, lon = np.float32(lat), np.float32(lon)

        remainder = i_%500
        if remainder == 0:
            print('')
            print(f'lat: {lat:.2f}, lon: {lon:.2f}, count: {i_}', end='')
        print('...', end='')


        # Get training & testing data:
        UE101_df = train_dss[0].sel(lat=lat, lon=lon, method='nearest')[features_+targets].to_dataframe()[features_+targets]
        UE102_df = train_dss[1].sel(lat=lat, lon=lon, method='nearest')[features_+targets].to_dataframe()[features_+targets]
        train_df = UE101_df.append(UE102_df)
        test_df = test_ds.sel(lat=lat, lon=lon, method='nearest')[features_+targets].to_dataframe()[features_+targets]

        # Get mean_x_num, std_x_num (mean_y, std_y for TSA):
        _, mean_x_num, std_x_num, train_y = prepare_input(train_df, targets, features_, is_train=True)
        if var == 'TSA':
            # Standardize y
            _, mean_y, std_y = standardize(train_y)  
        del UE101_df, UE102_df, train_df, test_df
        
        # Get input data:
        input_df = ds.sel(lat=lat, lon=lon, method='nearest')[features].to_dataframe()[features]
        
        # Prepare input data:
        apply_x, mean_x_num, std_x_num = prepare_input_apply(input_df, features, mean_x_num=mean_x_num, std_x_num=std_x_num)

        # Load model
        model = tf.keras.models.load_model(f'{savemodel_dir}/{var}_NN_{i_}_lat{lat:.2f}_lon{lon:.2f}')

        # Predict test_y
        apply_y_pred = model.predict(apply_x)

        if var == 'TSA':
            # de-standardize y
            apply_y_pred = destandardize(apply_y_pred, mean = mean_y, std = std_y)

        # Save NN predictions
        pred_da.loc[:, lat, lon] = apply_y_pred.squeeze()

        # Delete model
        del model

    print('')
    print(f'{var} model predict complete. Total models: {i_}.')
    print(f'Running time: {datetime.now() - start_time}')

    pred_da.to_netcdf(testpred_path)
    print('Apply preds saved:\t', testpred_path)

    return pred_da
