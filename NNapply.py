import importlib
import UrbanEnergyEmulator.NNfuncs
importlib.reload(UrbanEnergyEmulator.NNfuncs)
from UrbanEnergyEmulator.NNfuncs import *
# import matplotlib
# matplotlib.use('pdf')
# import matplotlib.pyplot as plt

import os

######################## CHANGE THIS SECTION ###########################

scenario = 'ssp245' # Choose from 'ssp245', 'ssp370', and 'ssp585'

train_mems = ['101', '102'] # denotes the members of CESM2 runs used for training 
test_mem = '103' # denotes the member of CESM2 run used for testing

models=[ # needs to be updated according to the scenario, based on Supplementary Table 1
    'ACCESS-CM2', 'ACCESS-ESM1-5', 'AWI-CM-1-1-MR', 'CNRM-CM6-1-HR', 
    'EC-Earth3', 'BCC-CSM2-MR', 'CMCC-CM2-SR5','FGOALS-f3-L', 'FGOALS-g3',
    'GFDL-ESM4', 'GISS-E2-1-G', 'GFDL-CM4',
    #'CNRM-ESM2-1', 'HadGEM3-GC31-MM', --> these two don't have ssp245
    'EC-Earth3-Veg', 'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'KACE-1-0-G', 
    'MIROC6', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM', 'FIO-ESM-2-0', 
    'NESM3'
        ]

targets = ['URBAN_HEAT_kJ'] # Choose from 'URBAN_HEAT_kJ', 'URBAN_AC_kJ', and 'TSA_U'

cesm2_file_path = '' # Directory where CESM2 training & testing data files are saved
saved_model_path = '' # Root directory to saved NN models
output_path = '' # Path where you'd like to save the results

########################################################################
# features = ['WIND', 'QBOT', 'TBOT', 'FLDS', 'FSDS', 'PRCP', 'month'] # feature variables 
features = ['wind', 'huss', 'tas', 'rlds', 'rsds', 'pr', 'month']


print('################################################ SET UP ################################################')
members = ['101', '102', '103']
target = targets[0]
var = met_var_dict[target]
compset = f'B{scenario.upper()}cmip6'
case_id = case_var_dict[target]+case_num_dict[compset]

print('Run on '+date.today().strftime("%Y-%m-%d"))
print(f'Case: {case_id}')
print(f'Target: {target}')
print(f'Scenario: {scenario}')

# Directory to retrieve saved NN models from:
savemodel_dir = saved_model_path + f'CASE{case_id}/{var}_kJ_NN_saved_CASE{case_id}'

# Load CESM data
ds_list = []
for mem in members:
    ds = xr.open_dataset(f'{cesm2_file_path}/{compset}.{mem}.h0.URBAN_ENERGY.201501-209912.nc', engine='netcdf4')
    ds.close()
    ds_list.append(ds)
URBAN_ENERGY_101, URBAN_ENERGY_102, URBAN_ENERGY_103 = ds_list
del ds_list

# Metrics file from training
metpath = f'.../CASE{case_id}/{var}_kJ_XGB_metrics_TEST_{test_mem}_CASE{case_id}.csv'
met = pd.read_csv(metpath).set_index(['lat', 'lon'])

# Apply NN to each model:
print('################################################ APPLY ################################################')
for esm in models:
    print(f'************************ Open file for {esm} ***************************')
    # Load ESM data:
    dir_out = output_path+esm+'/'
    ds = xr.open_dataset(dir_out+f'{esm}_{scenario}_URBAN_ENERGY_201501-209912.nc')
    ds.close()
    
    # Set up pred_da:
    testpred_path = dir_out+f'{esm}_{scenario}_{target}_NNpred_CASE{case_id}.201501-209912.nc'
    if os.path.isfile(testpred_path):
        pred_da = xr.open_dataarray(testpred_path)
        pred_da.close()
    else:
        init_shape = (URBAN_ENERGY_101.time.shape[0], 192, 288)
        init_data = np.full(init_shape, np.nan, dtype=np.float32) # a time, lat, lon (1020, 192, 288) array
        to_da = {'coords': {'lon': {'dims': 'lon', 'data': URBAN_ENERGY_101.lon.values,
                              'attrs': {'long_name': 'coordinate longitude', 'units': 'degrees_east'}},
                       'lat': {'dims': 'lat', 'data': URBAN_ENERGY_101.lat.values,
                              'attrs': {'long_name': 'coordinate latitude', 'units': 'degrees_north'}},
                       'time': {'dims': 'time', 'data': URBAN_ENERGY_101['time_corr'].values,
                              'attrs': {'long_name': 'time (corrected)', 'bounds': 'time_bounds'}}},
             'attrs': attrs_dict[var],
             'dims': ('time', 'lat', 'lon'),
             'data': init_data,
             'name': target + '_pred'}
        pred_da = xr.DataArray.from_dict(to_da)
        pred_da.attrs['is_training_data'] = 'False'
    print(f'Predicted {var} for {esm} {scenario} will be saved at:\n\t', testpred_path)
    
    print(f'************************ NN emulators: Apply {esm} ***************************')
    pred_da = apply(var, ds, features, targets, pred_da, met, savemodel_dir, testpred_path)
    
    print('')


