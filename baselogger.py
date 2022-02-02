#
# Programmed by Ashish Kumar (Senior Data Scientist) in year 2022 
# For any queries contact ashish.kumar@mail.mcgill.ca
#
'''
This module contains and initializes the logging capabilities
'''
## Python libraries
import logging
import os


## Internal libraries
import constants as gv

## Code
name_of_file_to_save = 'progress.log'
logger = logging

def setup_log_file_name(log_file_path=gv.log_file_path)->None:
    if not os.path.isdir(log_file_path):
        os.makedirs(log_file_path)
    logger.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S', filename=log_file_path+name_of_file_to_save, filemode='w')


