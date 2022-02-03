#
# Programmed by Ashish Kumar (Senior Data Scientist) in year 2022 
# For any queries contact ashish.kumar@mail.mcgill.ca
#

'''
This module contains the class that reads and writes the data
'''
## Python libraries
from typing import Tuple
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,LabelEncoder

## Internal libraries
import constants as gv
from baselogger import logger


## Code


class Data():

    # Data class have read, preprocess, and write functionality

    def __init__(self, file_path=gv.input_data_path, file_name=gv.input_file_name) -> None:
        self.input_data_path = file_path
        self.input_file_name = file_name
        self.input_path = self.input_data_path+self.input_file_name
        self.scaler = StandardScaler()
        self.feature_encoder = OrdinalEncoder()
        self.target_encoder = LabelEncoder()
        info = 'Initializing data reading and writing class'
        if gv.debug_level>=gv.major_details_print:
            print(info)
        logger.info(info)

        try:
            os.path.isfile(self.input_path)
            info = 'Input file located at {}'.format(self.input_path)
            if gv.debug_level>=gv.major_details_print:
                print(info)
            logger.info(info)

        except FileNotFoundError:
            info ='Cannot find the excel file. Please provide correct file name and path'
            if gv.debug_level>=gv.major_details_print:
                print(info)
            logger.info(info)

            info = 'Current file path is {} and file name is {}'.format(self.input_data_path,self.input_file_name)
            if gv.debug_level>=gv.major_details_print:
                print(info)
            logger.info(info)

    def read_data(self,)->object:
        info = 'Reading input data from file {}'.format(self.input_path)
        if gv.debug_level>=gv.minor_details_print:
            print(info)
        logger.info(info)
        
        try:
            data = pd.read_csv(self.input_path,index_col=0).dropna() # reading data, setting first column as index, and dropping na rows
            desc = data.describe(include='all')
            info=desc.to_string()
            self.write_data(data=desc,name='input_data_description',save_path=gv.output_data_path+gv.output_path_for_inputs)
            if gv.debug_level>=gv.major_details_print:
                print(info)
            logger.info(info)
            return data
        
        except FileNotFoundError:
            info ='Cannot find the excel file. Please provide correct file name'
            if gv.debug_level>=gv.major_details_print:
                print(info)
            logger.info(info)

            info = 'Current file path is {}'.format(self.input_data_path)
            if gv.debug_level>=gv.major_details_print:
                print(info)
            logger.info(info)

    def preprocess_data(self,data=None)->object:
        info = 'Preprocessing data'
        if gv.debug_level>=gv.minor_details_print:
            print(info)
        logger.info(info)
        try:
            data[gv.numeric_feature_column_name]=data[gv.numeric_feature_column_name].replace('[^0123456789\.]','',regex=True) # removing non-numeric values from numeric columns
            data[gv.numeric_feature_column_name]=data[gv.numeric_feature_column_name].apply(pd.to_numeric,errors='coerce') # converting numeric columns to numeric
            data = data.dropna() # finally dropping the rows that don't have all values
            desc = data.describe(include='all')
            info=desc.to_string()
            self.write_data(data=desc,name='preprocessed_data_description',save_path=gv.output_data_path+gv.output_path_for_inputs)
            if gv.debug_level>=gv.minor_details_print:
                print(info)
            logger.info(info)
            return data
        except Exception as e:
            info = e
            if gv.debug_level>=gv.major_details_print:
                print(info)
            logger.info(info) 

    def split_data_train_test(self,proportion=gv.test_prop,features=[],target=[]) -> Tuple[list,list,list,list,list,list]:
        info = 'Splitting data into train and test set'
        if gv.debug_level>=gv.minor_details_print:
            print(info)
        logger.info(info)
        try:
            self.splitter = StratifiedShuffleSplit(n_splits=1,test_size=proportion,random_state=0)
            for train_idx, test_idx in self.splitter.split(features,target):
                X_train = features[train_idx]
                Y_train = target[train_idx]
                X_test = features[test_idx]
                Y_test = target[test_idx]
            return X_train,Y_train,X_test,Y_test, train_idx, test_idx
        except Exception as e:
            info = e
            if gv.debug_level>=gv.major_details_print:
                print(info)
            logger.info(info) 

    def balance_dataset(self,features=[],target=[])->Tuple[list,list]:

        raise NotImplementedError

    def prepare_features_and_target(self,data=None)->Tuple[list,list]:
        info = 'Preparing features and target'
        if gv.debug_level>=gv.minor_details_print:
            print(info)
        logger.info(info)

        try:
            features = data[gv.features_column_name].values
            target = data[gv.target_column_name].values
            return features,target
        except Exception as e:
            info = e
            if gv.debug_level>=gv.major_details_print:
                print(info)
            logger.info(info) 

    def encode_features_and_target(self,data=None)->object:
        info = 'Encoding features and target'
        if gv.debug_level>=gv.minor_details_print:
            print(info)
        logger.info(info)

        try:
            info = 'Encoding features'
            if gv.debug_level>=gv.minor_details_print:
                print(info)
            logger.info(info)
            data_copied = data.copy(deep=True)
            data_copied[gv.categorical_feature_column_name]=self.feature_encoder.fit_transform(data_copied[gv.categorical_feature_column_name])
            info = self.feature_encoder.categories_
            if gv.debug_level>=gv.minor_details_print:
                print(info)
            logger.info(info)
            info = 'Encoding targets'
            if gv.debug_level>=gv.minor_details_print:
                print(info)
            logger.info(info)
            data_copied[gv.target_column_name]=self.target_encoder.fit_transform(data_copied[gv.target_column_name])
            info = self.target_encoder.classes_
            if gv.debug_level>=gv.minor_details_print:
                print(info)
            logger.info(info)
            self.write_data(data=data_copied,name='encoded_data',save_path=gv.output_data_path+gv.output_path_for_inputs)
            return data_copied
        except Exception as e:
            info = e
            if gv.debug_level>=gv.major_details_print:
                print(info)
            logger.info(info) 

    def scale_features(self,X=[],fit=False)->list:
        try:
            if fit:
                info = 'Scaling train features'
                X_scaled = self.scaler.fit_transform(X)
            else:
                info = 'Scaling test features'
                X_scaled=self.scaler.transform(X)
            if gv.debug_level>=gv.minor_details_print:
                print(info)
            logger.info(info)
            return X_scaled
        except Exception as e:
            info = e
            if gv.debug_level>=gv.major_details_print:
                print(info)
            logger.info(info)

    def inverse_scale_features(self,X=[])->list:
        info = 'Inverse scaling the features'
        if gv.debug_level>=gv.minor_details_print:
            print(info)
        logger.info(info)

        raise NotImplementedError
    
    def write_data(self,data=None,name=None,save_path=None)->None:
        info = 'Writing data {} to path {}'.format(name,save_path)
        if gv.debug_level>=gv.minor_details_print:
            print(info)
        logger.info(info)
        try:
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            data.to_csv(save_path+name+'.csv', index=True)

        except Exception as e:
            info = e
            if gv.debug_level>=gv.major_details_print:
                print(info)
            logger.info(info)  

