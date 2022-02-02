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