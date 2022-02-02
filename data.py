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
from baselogger import logger
import os


## Internal libraries
import constants as gv


## Code


class Data():

    # Data class have read, preprocess, and write functionality

    def __init__(self, file_path=gv.input_data_path, file_name=gv.input_file_name) -> None:
        self.input_data_path = file_path
        self.input_file_name = file_name
        self.input_path = self.input_data_path+self.input_file_name

    def read_data(self,)->object:

        pass

    def preprocess_data(self,data=None)->object:


        pass


    def split_data_train_test(self,propotion=None,features=[],target=[]) -> Tuple[list,list,list,list]:

        pass

    
    def write_data(self,data=None,name=None,save_path=None)->None:


        pass

    def balance_dataset(self,features=[],target=[])->Tuple[list,list]:

        raise NotImplementedError


    def prepare_features_and_target(self,data=None)->Tuple[list,list]:


        pass


    def generate_column_wise_description(self,data=None)->None:


        pass