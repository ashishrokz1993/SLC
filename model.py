#
# Programmed by Ashish Kumar (Senior Data Scientist) in year 2022 
# For any queries contact ashish.kumar@mail.mcgill.ca
#
'''
This module contains the class that creates/trains/test the supervised learning classification model
'''

## Python libraries
from copy import deepcopy
import numpy as np
import time
from typing import Tuple
import os
import warnings
warnings.filterwarnings('ignore')
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier,GradientBoostingClassifier
from sklearn.base import ClassifierMixin
from sklearn.utils import all_estimators
from xgboost import XGBClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pandas as pd

## Internal libraries
import constants as gv
from baselogger import logger

## Code

class SupervisedClassificationModel():

    def __init__(self,target_categories=[]) -> None:
        info = 'Initializing supervised learning model class'
        if gv.debug_level>=gv.major_details_print:
            print(info)
        logger.info(info)
        columns_df =['Method','AverageAccuracy']
        for c in target_categories:
            columns_df.append('Class_'+c+': Precision, Recall, F1-Score')
        columns_df.append('Macro Avg: Precision, Recall, F1-Score')
        columns_df.append('Weighted Avg: Precision, Recall, F1-Score')
        self.stats_test = pd.DataFrame(columns=columns_df).set_index('Method')
        self.stats_train = pd.DataFrame(columns=columns_df).set_index('Method')

    def get_classifier_list(self,)->None:
        info = 'Testing the trained model'
        if gv.debug_level>=gv.minor_details_print:
            print(info)
        logger.info(info)

        try:
            classifiers=[est for est in all_estimators() if issubclass(est[1], ClassifierMixin)]
            list_c= []
            for c in classifiers:
                list_c.append(c)
            info = list_c
            if gv.debug_level>=gv.minor_details_print:
                print(info)
            logger.info(info) 
        
        except Exception as e:
            info = e
            if gv.debug_level>=gv.major_details_print:
                print(info)
            logger.info(info)


    def train(self,x_train = [], y_train = [],x_test=[],y_test=[],target_encoder=None)-> Tuple[list,list]:
        info = 'Training SL classification model'
        if gv.debug_level>=gv.minor_details_print:
            print(info)
        logger.info(info)

        try:
            params = {'n_neighbors': 3,
                            'kernel': 'linear',
                            'c_linear': 0.025,
                            'c_rbf': 1,
                            'gamma': 2,
                            'max_depth': 5,
                            'n_estimators': 10,
                            'max_features': 1,
                            'alpha': 1,
                            'max_iter':1000}
            nnc = KNeighborsClassifier(n_neighbors=params['n_neighbors'])
            svc = SVC(kernel=params['kernel'],C=params['c_linear'])
            dtc = DecisionTreeClassifier(max_depth=params['max_depth'])
            rfc =RandomForestClassifier(max_depth=params['max_depth'],n_estimators=params['n_estimators'],max_features=params['max_features'])
            xgbc = XGBClassifier()
            etc = ExtraTreesClassifier(max_depth=params['max_depth'],n_estimators=params['n_estimators'],max_features=params['max_features'])
            bc = BaggingClassifier(base_estimator=nnc,max_features=params['max_features'])
            gbc = GradientBoostingClassifier(n_estimators=params['n_estimators'],max_depth=params['max_depth'])
            supervised_algorithms_and_parameters= (
                ('Nearest Neighbors',nnc,{'n_neighbors':[2,4,6,8],}),
                ('Gradient Boosting', gbc, {'n_estimators':[2,4,6,8,10],'max_depth':[3,4,5,6]}),
                ('Random Forest', rfc,{'n_estimators':[2,4,6,8,10],'max_depth':[3,4,5,6]}),
                ('Support Vector Machines', svc,{'gamma':[0.0001,0.001,0.01,0.1,1,10,100],'C':[0.1,1,5,10,50,100]}),
                ('Xtreme Gradient Boosting',xgbc, {
                                                    'max_depth':[3,4,5,6],
                                                    'n_estimators':[2,4,6,8,10],
                                                }),
                ('Decision Tree', dtc, {'max_depth':[3,4,5,6]}),
                ('ExtraTree',etc,{'n_estimators':[2,4,6,8,10]}),
                ('Bagging',bc, {'n_estimators':[2,4,6,8,10]}),

            )
            best_model_name = None
            best_model_param =None
            best_model = None
            best_score=0
            for name, algorithm,parameters in supervised_algorithms_and_parameters:
                t0= time.time()
                info = 'Working on algorithm {}'.format(name)
                if gv.debug_level>=gv.minor_details_print:
                    print(info)
                logger.info(info)
                clf = GridSearchCV(estimator=algorithm,param_grid=parameters,scoring='average_precision')
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="the number of connected components of the " +
                        "connectivity matrix is [0-9]{1,2}" +
                        " > 1. Completing it to avoid stopping the tree early.",
                        category=UserWarning)
                    warnings.filterwarnings(
                        "ignore",
                        message="Graph is not fully connected, spectral embedding" +
                        " may not work as expected.",
                        category=UserWarning)
                    clf.fit(x_train,y_train)
                t1= time.time()
                
                info = 'Total training and tuning time {}'.format(t1-t0)
                if gv.debug_level>=gv.minor_details_print:
                    print(info)
                logger.info(info)

                info = 'Best params: {}'.format(clf.best_params_)
                if gv.debug_level>=gv.minor_details_print:
                    print(info)
                logger.info(info)
                
                info = 'Best score: {}'.format(clf.best_score_)
                if gv.debug_level>=gv.minor_details_print:
                    print(info)
                logger.info(info)
                
                means = clf.cv_results_['mean_test_score']
                stds = clf.cv_results_['std_test_score']
                list_stats = [['Mean     Standard Deviation      Parameters']]
                for m,s,p in zip(means, stds, clf.cv_results_['params']):
                    list_stats.append(['{:.3f}'.format(m),'(+/- {:.3f})'.format(s*2),p ])
                info=list_stats
                if gv.debug_level>=gv.minor_details_print:
                    print(info)
                logger.info(info)
                if gv.save_and_use_best:
                    if clf.best_score_>=best_score:
                        best_score=clf.best_score_
                        best_model_name=name
                        best_model_param=clf.best_params_
                        best_model = deepcopy(clf)
                else:
                    if name==gv.name_of_model_to_save: # Modified this to introduce saving SVM model because its a simpler model given the amount of data we have
                        best_score=clf.best_score_
                        best_model_name=name
                        best_model_param=clf.best_params_
                        best_model = deepcopy(clf)
                
                y_train_pred = clf.predict(x_train)
                report_train = classification_report(y_train,y_train_pred,output_dict=True)

                y_test_pred = clf.predict(x_test)
                report_test=classification_report(y_test,y_test_pred,output_dict=True)

                self.stats_test.loc[name,'AverageAccuracy']=report_test['accuracy']
                self.stats_test.loc[name,'Weighted Avg: Precision, Recall, F1-Score'] = "{:.2f}".format(report_test['weighted avg']['precision'])+' , '+"{:.2f}".format(report_test['weighted avg']['recall'])+' , '+"{:.2f}".format(report_test['weighted avg']['f1-score'])
                self.stats_test.loc[name,'Macro Avg: Precision, Recall, F1-Score'] = "{:.2f}".format(report_test['macro avg']['precision'])+' , '+"{:.2f}".format(report_test['macro avg']['recall'])+' , '+"{:.2f}".format(report_test['macro avg']['f1-score'])
                for key in report_test.keys():

                    if key != 'accuracy' and key!='macro avg' and key!='weighted avg':
                            class_code = target_encoder.inverse_transform([int(key)])
                            column_class = 'Class_'+class_code+': Precision, Recall, F1-Score'
                            column_class_p = report_test[key]['precision']
                            column_class_r = report_test[key]['recall']
                            column_class_f = report_test[key]['f1-score']
                            final_val = "{:.2f}".format(column_class_p)+' , '+"{:.2f}".format(column_class_r)+' , '+"{:.2f}".format(column_class_f)
                            self.stats_test.loc[name,column_class]=final_val
                
                self.stats_train.loc[name,'AverageAccuracy']=report_train['accuracy']
                self.stats_train.loc[name,'Weighted Avg: Precision, Recall, F1-Score'] = "{:.2f}".format(report_train['weighted avg']['precision'])+' , '+"{:.2f}".format(report_train['weighted avg']['recall'])+' , '+"{:.2f}".format(report_train['weighted avg']['f1-score'])
                self.stats_train.loc[name,'Macro Avg: Precision, Recall, F1-Score'] = "{:.2f}".format(report_train['macro avg']['precision'])+' , '+"{:.2f}".format(report_train['macro avg']['recall'])+' , '+"{:.2f}".format(report_train['macro avg']['f1-score'])
                for key in report_train.keys():
                    if key != 'accuracy' and key!='macro avg' and key!='weighted avg':
                            class_code = target_encoder.inverse_transform([int(key)])
                            column_class = 'Class_'+class_code+': Precision, Recall, F1-Score'
                            column_class_p = report_train[key]['precision']
                            column_class_r = report_train[key]['recall']
                            column_class_f = report_train[key]['f1-score']
                            final_val = "{:.2f}".format(column_class_p)+' , '+"{:.2f}".format(column_class_r)+' , '+"{:.2f}".format(column_class_f)
                            self.stats_train.loc[name,column_class]=final_val
            
            self.train_model = best_model.best_estimator_
            self.train_model_name = best_model_name
            self.train_model_param = best_model_param
            self.save_trained_model()
            y_train_pred = self.train_model.predict(x_train)
            y_test_pred = self.train_model.predict(x_test)
            info = 'Best overall model is {} with parameters {} and score {}'.format(self.train_model_name,self.train_model_param,best_score)
            if gv.debug_level>=gv.minor_details_print:
                print(info)
            logger.info(info)
            return y_train_pred,y_test_pred
        except Exception as e:
            info = e
            if gv.debug_level>=gv.major_details_print:
                print(info)
            logger.info(info)

    def save_trained_model(self)->None:
        info = 'Saving trained SL model'
        if gv.debug_level>=gv.minor_details_print:
            print(info)
        logger.info(info)

        try:
            pickle.dump(self.train_model,open(gv.output_data_path+gv.output_path_for_outputs+self.train_model_name+'.pkl','wb'))
        
        except Exception as e:
            info = e
            if gv.debug_level>=gv.major_details_print:
                print(info)
            logger.info(info)

    def test_model(self,x_test = [])->None:
        info = 'Testing the trained model'
        if gv.debug_level>=gv.minor_details_print:
            print(info)
        logger.info(info)

        try:
            return self.train_model.predict(x_test)

        except Exception as e:
            info = e
            if gv.debug_level>=gv.major_details_print:
                print(info)
            logger.info(info)
