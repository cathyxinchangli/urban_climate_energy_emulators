import importlib
import UrbanEnergyEmulator.NNfuncs
importlib.reload(UrbanEnergyEmulator.NNfuncs)
from UrbanEnergyEmulator.NNfuncs import *
print('Updated on '+date.today().strftime("%Y-%m-%d"))
import os
import matplotlib.pyplot as plt

######################## CHANGE THIS SECTION ###########################

scenario = 'ssp245' # Choose from 'ssp245', 'ssp370', and 'ssp585'

train_mems = ['101', '103'] # denotes the members of CESM2 runs used for training 
test_mem = '102' # denotes the member of CESM2 run used for testing

targets = ['URBAN_AC_kJ'] # one of ['URBAN_HEAT_kJ'], ['URBAN_AC_kJ'], ['TSA_U']

verbose_code = 0 # verbose behavior of NN training: 0 - silent, 1 - progress bar, 2 - one line per epoch
make_plots = False # whether to make plots for training & testing performance

cesm2_file_path = '' # Directory where CESM2 training & testing data files are saved
saved_model_path = '' # Root directory to saved NN models
output_path = '' # Path where you'd like to save the results

########################################################################


print('################################################ SET UP ################################################')
members = ['101', '102', '103']
features = ['WIND', 'QBOT', 'TBOT', 'FLDS', 'FSDS', 'PRCP', 'month']
target = targets[0]
var = met_var_dict[target]
compset = f'B{scenario.upper()}cmip6'
case_id = case_var_dict[target]+case_num_dict[compset]

print('Run on '+date.today().strftime("%Y-%m-%d"))
print(f'Case: {case_id}')
print(f'Target: {target}')
print(f'Scenario: {scenario}')

# Load CESM data
ds_list = []
for mem in members:
    ds = xr.open_dataset(f'{cesm2_file_path}/{compset}.{mem}.h0.URBAN_ENERGY.201501-209912.nc', engine='netcdf4')
    ds.close()
    ds_list.append(ds)
URBAN_ENERGY_101, URBAN_ENERGY_102, URBAN_ENERGY_103 = ds_list
del ds_list

# Load csv of urban grid cells
gridcells = pd.read_csv(f'{cesm2_file_path}/urban_gridcells.csv').set_index(['lat', 'lon'])

# Get hyperparameters
learning_rate = hp_dict[var]['learning_rate']
l1 = hp_dict[var]['l1']
l2 = hp_dict[var]['l2']
output_activation = hp_dict[var]['output_activation']
batch_size = hp_dict[var]['batch_size']
num_epochs = hp_dict[var]['epochs']

# Train NN models by looping through all grid cells
print('############################################## TRAIN ################################################')

start_time = datetime.now()
for i, (lat, lon) in enumerate(gridcells.index):
    lat, lon = np.float32(lat), np.float32(lon)
    print(f'{i}: ({lat}, {lon})')
    
    # Get training & testing data:
    UE101_df = URBAN_ENERGY_101.sel(lat=lat, lon=lon, method='nearest')[features+targets].to_dataframe()[features+targets]
    UE102_df = URBAN_ENERGY_102.sel(lat=lat, lon=lon, method='nearest')[features+targets].to_dataframe()[features+targets]
    UE103_df = URBAN_ENERGY_103.sel(lat=lat, lon=lon, method='nearest')[features+targets].to_dataframe()[features+targets]

    train_df = eval(f'UE{train_mems[0]}_df').append(eval(f'UE{train_mems[1]}_df'))
    valtest_df = eval(f'UE{test_mem}_df')
    
    # Randomly select 30% gridcells for validation
    msk = np.random.rand(len(valtest_df)) < .3
    val_df = valtest_df[msk]
    test_df = valtest_df[~msk]
    
    # Prepare training data for input
    train_x, mean_x_num, std_x_num, train_y = prepare_input(train_df, targets, features, is_train=True)
    val_x, _, _, val_y = prepare_input(val_df, targets, features, is_train=False, mean_x_num=mean_x_num, std_x_num=std_x_num)

    if var == 'TSA':
        # Standardize y
        train_y, mean_y, std_y = standardize(train_y)
        val_y, _, _ = standardize(val_y, mean = mean_y, std = std_y)

    # Build NN
    model = create_model(learning_rate, l1, l2, output_activation=output_activation)
    
    # Train NN
    history = model.fit(
        train_x,
        train_y,
        validation_data = (val_x, val_y),
        verbose=verbose_code,
        batch_size=batch_size,
        epochs=num_epochs, 
        shuffle=True,
    ) 

    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    
    if make_plots:
        # Training performance
        plt.xlabel("Epoch")
        plt.ylabel("Root Mean Squared Error")
        plt.plot(epochs, hist.root_mean_squared_error, label="train rmse")
        plt.plot(epochs, hist.val_root_mean_squared_error, label="val rmse")
        plt.legend()
        plt.show()
    
    # Prepare testing data for input
    test_x, _, _, test_y = prepare_input(test_df, targets, features, is_train=False, mean_x_num=mean_x_num, std_x_num=std_x_num)

    # Predict test_y
    test_y_pred = model.predict(test_x)

    if var == 'TSA':
        # de-standardize y
        test_y_pred = destandardize(test_y_pred, mean = mean_y, std = std_y)

    # Evaluate test performance
    rsq, rmse, rsq_xgb, rmse_xgb = evaluate_performance(test_y, test_y_pred, verbose=True)

    if make_plots:
        # Plot truths and predictions from NN
        fig, axes = plt.subplots(1, 1, figsize=(20,4))
        test_df[targets[0]].plot(ax=axes, color='gray', linestyle='--', linewidth=2, alpha=0.3, label=f'{targets[0]} CESM2')
        axes.plot(test_df.index.astype(str), test_y_pred, '.', color='k', alpha=.8, label=f'{targets[0]} NN (rsq={round(rsq, 3)}, rmse={round(rmse, 3)})')
        axes.legend()
        plt.show()
    
    # Save model
    tf.keras.models.save_model(model.model, saved_model_path+f'CASE{case_id}/{var}_kJ_NN_saved_CASE{case_id}')
    
    # Delete model
    del model

print(f'Running time: {datetime.now()-start_time}')