This repository contains the code for exploratory data analysis of people data and training a supervised learning model for classifying targets

Programmed by Ashish Kumar (ashish.kumar@mail.mcgill.ca)

### Compile and Run Instructions

1. Install [Anaconda](https://www.anaconda.com/products/individual)
2. Install [VSCode](https://code.visualstudio.com/download)
3. Install python for VSCode
4. Start Anaconda and run the following commands: conda create --name slcpd python=3.7.6
5. Open VSCode and navigate to the project directory
6. Choose the environment as slcpd
7. Now go to terminal and run the command pip install -r requirements.txt
8. Place the data in the data folder
9. Run main.py. It should run without any errors.


### Features include

1. Loading the data using a data class
2. Exploratory analysis and preprocessing methods for the data to remove undesirable inputs and convert inputs and output to required format
3. Model building class for supervised learning classification model training, hyper-parameter optimization and evaluation
4. Plotting class for plotting static and dynamic graphs input data and results
5. Logging class for documenting progress


### Modelling Steps
1. Load the input data
2. Preprocess the data
    1. Remove blanks
    2. Replace non-float to float in columns that are float
    3. Remove rows that have not valid values
3. Encode the features and targets
4. Divide the data into training and testing set
5. Scaler fit and transform for train set features
6. Train supervised learning classification model while hyper parameter tuning using grid search
7. Plot input data graphs and predicted labels vs true labels using UMAP

### Assumptions
1. Float columns cannot have non-float values
2. The data is not time-series so we don't have to worry about stationarity 
3. The data rows seems to represent individual independent events
4. The data is little imbalance (not so much 30~40% no label vs 70~60% yes label). Balancing was tried but it made the result worse by introducing noise
5. Simpler model will result in a better and generalized model since the number of training samples are not a lot
6. The hyper-parameters like depth of tree, number of estimators are tuned within low limits to avoid overfitting
7. The data drifts that might happen during testing because of categorical features will be dealt via data-drift features (and re-training during deployment)
8. The features which have float values are assumed to have only float values and the preprocessing was done to remove any non-float values for those features


