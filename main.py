#
# Programmed by Ashish Kumar (Senior Data Scientist) in year 2022 
# For any queries contact ashish.kumar@mail.mcgill.ca
#

'''
This is the main module that uses the rest of the modules to perform the task
'''
## Python libraries
import os
from imblearn.over_sampling import SMOTE



## Internal libraries
import constants as gv
from baselogger import logger, setup_log_file_name
from data import Data
from plotter import Graphs
from model import SupervisedClassificationModel


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

plotter.plot_column_wise_description(data=data,encoded=False) # Plotting graphs for column description

data=data.drop(columns=gv.features_to_drop_categorical+gv.features_to_drop_numeric) # features were dropped because there were highly correlated

# Creating SL model object
clf = SupervisedClassificationModel(target_categories=data[gv.target_column_name].unique())

data_enc = input.encode_features_and_target(data=data) # Encoding features and target

plotter.plot_column_wise_description(data=data_enc,encoded=True) # Plotting graphs for column description

features, target = input.prepare_features_and_target(data=data_enc) # Generating list of features and targets

x_train, y_train, x_test, y_test,train_ids, test_ids = input.split_data_train_test(features=features,target=target) # Splitting into train and test

x_train = input.scale_features(X=x_train,fit=True) # Scaling train features

x_test=input.scale_features(X=x_test,fit=False) # Scaling test features

if gv.balance_dataset_training:
    ## Trying resampling to see if it improves results
    x_train_resampled, y_train_resampled = SMOTE().fit_resample(x_train,y_train)
    plotter.plotly_graphs_2d(x_before=x_train,y_before=y_train,y_before_encoded=input.target_encoder.inverse_transform(y_train),x_after=x_train_resampled,y_after=y_train_resampled,y_after_encoded=input.target_encoder.inverse_transform(y_train_resampled)) 
    y_train_pred,y_test_pred = clf.train(x_train=x_train_resampled,y_train=y_train_resampled,x_test=x_test,y_test=y_test,target_encoder=input.target_encoder)
else:
    y_train_pred,y_test_pred = clf.train(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,target_encoder=input.target_encoder)
input.write_data(data=clf.stats_train,name='train_stats',save_path=gv.output_data_path+gv.output_path_for_outputs)
input.write_data(data=clf.stats_test,name='test_stats',save_path=gv.output_data_path+gv.output_path_for_outputs)

final_features_columns=[i for i in gv.features_column_name if i not in gv.features_to_drop_categorical+gv.features_to_drop_numeric]
plotter.plot_feature_importance(clf=clf.train_model,x=x_train,y=y_train,columns=final_features_columns)
plotter.plotly_graphs(x=x_test,y_true_encoded=y_test,y_pred_encoded=y_test_pred,y_true=data[gv.target_column_name].values[test_ids],y_pred=input.target_encoder.inverse_transform(y_test_pred))



