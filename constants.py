#
# Programmed by Ashish Kumar (Senior Data Scientist) in year 2022 
# For any queries contact ashish.kumar@mail.mcgill.ca
#

'''
This module contains all the constants used through the program. This should never import any modules
'''


## Code


## Input and output file parameters
input_data_path = 'Data/'
input_file_name = '2021-10-19_14-11-08_val_candidate_data.csv'
output_data_path = 'Results/Files/'
output_graphs_path = 'Results/Plots/'
output_path_for_inputs = 'Inputs/'
output_path_for_outputs = 'Outputs/'
log_file_path = 'Logs/'
target_column_name = 'target'
num_features =13
features_column_name = ['feature_'+str(i) for i in range(num_features)]
numeric_feature_column_name = ['feature_'+str(i) for i in [2,5,6,7,8,9,11]]
categorical_feature_column_name = [i for i in features_column_name if i not in numeric_feature_column_name]

features_to_drop_categorical =['feature_12']
features_to_drop_numeric =['feature_11'] # removed because its a redundant feature (observed in correlation plot)



## Debugging options
no_print = 2
major_details_print = 1
minor_details_print=0 
debug_level = 1


## Supervised learning classification model parameters
test_prop = 0.7
save_and_use_best= False
name_of_model_to_save = 'Support Vector Machines'
balance_dataset_training = False


## Factors

default_dict_value = -99


