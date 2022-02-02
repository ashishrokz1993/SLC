#
# Programmed by Ashish Kumar (Senior Data Scientist) in year 2022 
# For any queries contact ashish.kumar@mail.mcgill.ca
#

'''
This is the main module that uses the rest of the modules to perform the task
'''
## Python libraries
import os
import numpy as np


## Internal libraries
import constants as gv
from baselogger import logger, setup_log_file_name
from data import Data
from plotter import Graphs


## Code

setup_log_file_name(gv.log_file_path) # Setting up log file path

info = 'Run started'
if gv.debug_level>=gv.major_details_print:
    print(info)
logger.info(info)


info = 'Run directory is {}'.format(os.getcwd())
if gv.debug_level>=gv.major_details_print:
    print(info)
logger.info(info)

# Creating data class object
input = Data()

# Creating a plotter object
plotter = Graphs()


data = input.read_data() # Reading input data

data = input.preprocess_data(data=data) # preprocessing data to remove non-desirable inputs

plotter.plot_column_wise_description(data=data) # Plotting graphs for column description

data = input.encode_features_and_target(data=data)

features, target = input.prepare_features_and_target(data=data)

x_train, y_train, x_test, y_test,train_ids, test_ids = input.split_data_train_test(features=features,target=target)

x_train = input.scale_features(X=x_train,fit=True)

x_test=input.scale_features(X=x_test,fit=False)

