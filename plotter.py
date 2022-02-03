#
# Programmed by Ashish Kumar (Senior Data Scientist) in year 2022 
# For any queries contact ashish.kumar@mail.mcgill.ca
#
'''
This module contains the class that plots all the relevant graphs
'''
## Python libraries
from base64 import encode
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio
import seaborn as sns
import umap.umap_ as umap
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import graphviz
from sklearn import tree


## Internal libraries
import constants as gv
from baselogger import logger
logger.getLogger('matplotlib.font_manager').disabled = True
logger.getLogger('PIL').setLevel(logger.WARNING)
logger.getLogger('numba').setLevel(logger.WARNING)


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
        plt.rcParams["figure.figsize"] = (60,20)
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

    def plotly_graphs(self,x=[],y_true_encoded=[],y_pred_encoded=[],y_true=[],y_pred=[])->None:
        info = 'Plotting 3d plots for test data'
        if gv.debug_level>=gv.minor_details_print:
            print(info)
        logger.info(info)
        try:
            umap_3d = umap.UMAP(n_components=3, init='random', random_state=0)
            proj_3d = umap_3d.fit_transform(x)
            fig_3d = make_subplots(rows=1,cols=2,subplot_titles=("True labels", "Predicted labels"),
                                    specs = [[{"type": "scatter3d"},{"type": "scatter3d"}]])
            fig_3d.add_trace(go.Scatter3d(x=proj_3d[:,0],y=proj_3d[:,1],z=proj_3d[:,2],mode='markers',
                                    marker_color=y_true_encoded, text=y_true,showlegend=False),row=1,col=1)
            fig_3d.add_trace(go.Scatter3d(x=proj_3d[:,0],y=proj_3d[:,1],z=proj_3d[:,2],mode='markers',
                                    marker_color=y_pred_encoded, text=y_pred,showlegend=False),row=1,col=2)
            fig_3d.update_traces(marker_size=5)
            fig_3d.update_layout(
                    showlegend=True,
                    uirevision= True,
                    title={
                        'text': 'UMAP plot',
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                    scene=dict(
                        xaxis=dict(
                                showline=True,
                                showgrid=False,
                                zeroline=False,
                                showticklabels=True,
                                #linecolor='black',
                                titlefont= dict(color= "black"),
                                title='Component 1',
                                #backgroundcolor  = 'rgb(40,40,40,0)'                       
                        ),
                        yaxis=dict(
                                showline=True,
                                showgrid=False,
                                #linecolor='black',
                                zeroline=False,
                                showticklabels=True,
                                titlefont= dict(color= "black"),
                                title='Component 2',        
                                #backgroundcolor  = 'rgb(40,40,40,0)'                  
                        ),
                        zaxis=dict(
                                showline=True,
                                showgrid=False,
                                #linecolor='black',
                                zeroline=False,
                                showticklabels=True,
                                titlefont= dict(color= "black"),
                                title='Component 3',    
                                #backgroundcolor  = 'rgb(40,40,40,0)'                      
                        ),
                    ),
                    scene2=dict(
                        xaxis=dict(
                                showline=True,
                                showgrid=False,
                                zeroline=False,
                                showticklabels=True,
                                #linecolor='black',
                                titlefont= dict(color= "black"),
                                title='Component 1',
                                #backgroundcolor  = 'rgb(40,40,40,0)'                       
                        ),
                        yaxis=dict(
                                showline=True,
                                showgrid=False,
                                #linecolor='black',
                                zeroline=False,
                                showticklabels=True,
                                titlefont= dict(color= "black"),
                                title='Component 2',        
                                #backgroundcolor  = 'rgb(40,40,40,0)'                  
                        ),
                        zaxis=dict(
                                showline=True,
                                showgrid=False,
                                #linecolor='black',
                                zeroline=False,
                                showticklabels=True,
                                titlefont= dict(color= "black"),
                                title='Component 3',    
                                #backgroundcolor  = 'rgb(40,40,40,0)'                      
                        ),
                    ),
            )
            fig_3d.write_html(self.graphs_path_outputs+'output.html')

        except Exception as e:
            info = e
            if gv.debug_level>=gv.major_details_print:
                print(info)
            logger.info(info) 

    def plotly_graphs_2d(self,x_before=[],y_before=[], y_before_encoded=[], x_after=[],y_after=[],y_after_encoded=[])->None:
        
        info = 'Plotting original and balanced dataset'
        if gv.debug_level>=gv.minor_details_print:
            print(info)
        logger.info(info)

        try:
            umap2d = umap.UMAP(n_components=2, init='random', random_state=0)
            features_umap_2d_before = umap2d.fit_transform(x_before)

            fig_umap_2d = make_subplots(rows=1,cols=2,subplot_titles=("Original Dataset", "Balanced Dataset"))
            
            fig_umap_2d.add_trace(go.Scatter(x=features_umap_2d_before[:,0],y=features_umap_2d_before[:,1],mode='markers',
                                    marker_color=y_before, text=y_before_encoded,showlegend=False),row=1,col=1)
            
            features_umap_2d_after = umap2d.fit_transform(x_after)                        
            fig_umap_2d.add_trace(go.Scatter(x=features_umap_2d_after[:,0],y=features_umap_2d_after[:,1],mode='markers',
                                    marker_color=y_after, text=y_after_encoded,showlegend=False),row=1,col=2)
            
            fig_umap_2d.update_layout(
                    showlegend=True,
                    uirevision= True,
                    title={
                        'text': 'UMAP plot',
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                    xaxis=dict(
                                showline=True,
                                showgrid=False,
                                zeroline=False,
                                showticklabels=True,
                                titlefont= dict(color= "black"),
                                title='Component 1',                    
                        ),
                    xaxis2=dict(
                                showline=True,
                                showgrid=False,
                                zeroline=False,
                                showticklabels=True,
                                titlefont= dict(color= "black"),
                                title='Component 1',                    
                        ),
                    yaxis=dict(
                                showline=True,
                                showgrid=False,
                                zeroline=False,
                                showticklabels=True,
                                titlefont= dict(color= "black"),
                                title='Component 2',                        
                        ),
                    yaxis2=dict(
                                showline=True,
                                showgrid=False,
                                zeroline=False,
                                showticklabels=True,
                                titlefont= dict(color= "black"),
                                title='Component 2',                        
                        ),
            )
            fig_umap_2d.write_html(self.graphs_path_inputs+'input.html')
        except Exception as e:
            info = e
            if gv.debug_level>=gv.major_details_print:
                print(info)
            logger.info(info)
    
    def plot_feature_importance(self,clf=None,x=[],y=[],columns=[])->None:
        info = 'Plotting feature importance for classifier {}'.format(clf)
        if gv.debug_level>=gv.minor_details_print:
            print(info)
        logger.info(info)
        try:
            result = permutation_importance(clf, x, y, n_repeats=10,random_state=42)
            perm_sorted_idx = result.importances_mean.argsort()
            tree_importance_sorted_idx = np.argsort(clf.feature_importances_)
            tree_indices = np.arange(0, len(clf.feature_importances_)) + 0.5
            fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(20, 20))
            ax1.barh(tree_indices,clf.feature_importances_[tree_importance_sorted_idx])
            ax1.set_yticks(tree_indices)
            ax1.set_xlabel('Score')
            ax1.set_ylabel('Feature')
            
            ax1.set_yticklabels(columns[tree_importance_sorted_idx])
            ax1.set_ylim((0, len(clf.feature_importances_)))
            ax1.set_title('Classifier Feature Importance')
            ax2.boxplot(result.importances[perm_sorted_idx].T, vert=False,
                        labels=columns[perm_sorted_idx])
            ax2.set_title('Permutation Feature Importance')
            ax2.set_xlabel('Score')
            ax2.set_ylabel('Feature')
            plt.suptitle('Feature Importance Graph')
            fig.tight_layout()
            
            plt.savefig(self.graphs_path_outputs+'feature_importance.png')
        except Exception as e:
            info = e
            if gv.debug_level>=gv.major_details_print:
                print(info)
            logger.info(info) 

    def plot_column_wise_description(self,data=None,encoded=True)->None:
        info = 'Plotting column wise description of the data'
        if gv.debug_level>=gv.minor_details_print:
            print(info)
        logger.info(info)

        try:
            if encoded:
                correlation = data.corr()
                sns.heatmap(correlation,xticklabels=correlation.columns,yticklabels=correlation)
                plt.tight_layout()
                plt.savefig(self.graphs_path_inputs+'correlation_matrix.png')
                plt.clf()
            else:
                fig, ax = plt.subplots(1, len(gv.categorical_feature_column_name+[gv.target_column_name]))
                fig.tight_layout()
                for i, categorical_feature in enumerate(data[gv.categorical_feature_column_name+[gv.target_column_name]]):
                    data[categorical_feature].value_counts().plot(kind="bar", ax=ax[i],rot=0).set_title(categorical_feature)
                fig.suptitle('Frequency Plot: Categorical Variables')
                fig.subplots_adjust(top=0.88)
                plt.savefig(self.graphs_path_inputs+'categorical_variables_histogram.png')
                plt.clf()

                fig, ax = plt.subplots(1, len(gv.numeric_feature_column_name))
                fig.tight_layout()
                data.hist(bins=50,ax=ax)
                fig.suptitle('Frequency Plot: Continuous Variables')
                fig.subplots_adjust(top=0.88)
                plt.savefig(self.graphs_path_inputs+'continuous_variables_histogram.png')
                plt.clf()

                fig, ax = plt.subplots(1, 1)
                fig.tight_layout()
                for i, categorical_feature in enumerate(data[gv.target_column_name]):
                    data[categorical_feature].value_counts().plot(kind="bar", ax=ax[i],rot=0).set_title(categorical_feature)
                fig.suptitle('Frequency Plot: Target')
                fig.subplots_adjust(top=0.88)
                plt.savefig(self.graphs_path_inputs+'target_histogram.png')
                plt.show()
                plt.clf()
                

        except Exception as e:
            info = e
            if gv.debug_level>=gv.major_details_print:
                print(info)
            logger.info(info) 

    def plot_decision_tree(self,clf=None,feature_name=[],target_categories = [],algorithm_name=None)->None:
        info = 'Plotting decision tree for SL methods that uses trees'
        if gv.debug_level>=gv.minor_details_print:
            print(info)
        logger.info(info)

        try:
            dot_data =tree.export_graphviz(decision_tree=clf, out_file=None,feature_names=feature_name,class_names=target_categories,filled=True,precision=2) 
            graph = graphviz.Source(dot_data)
            graph.render(self.graphs_path_outputs+'decision_tree_'+algorithm_name,format='png')
        
        except Exception as e:
            info = e
            if gv.debug_level>=gv.major_details_print:
                print(info)
            logger.info(info)
    