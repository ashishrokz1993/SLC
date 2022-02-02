#
# Programmed by Ashish Kumar (Senior Data Scientist) in year 2022 
# For any queries contact ashish.kumar@mail.mcgill.ca
#
'''
This module contains the class that plots all the relevant graphs
'''
## Python libraries
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio


## Internal libraries
import constants as gv
from baselogger import logger
logger.getLogger('matplotlib.font_manager').disabled = True
logger.getLogger('PIL').setLevel(logger.WARNING)

## Code

class Graphs():


    def __init__(self,graphs_path=gv.output_graphs_path,input_data_graphs_path=gv.output_path_for_inputs,output_data_graphs_path=gv.output_path_for_outputs) -> None:
        SMALL_SIZE = 30
        MEDIUM_SIZE = 45
        BIGGER_SIZE = 45
        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        plt.rcParams["figure.figsize"] = (20,20)
        #pio.kaleido.scope.default_width = 1000
        #pio.kaleido.scope.default_height = 1000

        self.graphs_path_inputs = graphs_path+input_data_graphs_path
        self.graphs_path_outputs=graphs_path+output_data_graphs_path
        info = 'Initializing visualization class object and setting up input and output data graphs path to {} and {}'.format(self.graphs_path_inputs,self.graphs_path_outputs)
        if gv.debug_level>=gv.major_details_print:
            print(info)
        logger.info(info)
        if not os.path.isdir(self.graphs_path_inputs):
            os.makedirs(self.graphs_path_inputs)
        if not os.path.isdir(self.graphs_path_outputs):
            os.makedirs(self.graphs_path_outputs)

    def matplotlib_graphs(self,)->None:

        pass

    def plotly_graphs(self,)->None:


        pass

    def plot_feature_importance(self,)->None:

        pass

    def plot_column_description(self,)->None:


        pass


    