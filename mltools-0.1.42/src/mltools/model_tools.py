import matplotlib.pyplot as plt
import math 
import numpy as np
import seaborn as sns
import pandas as pd

def plotModelGridError(model, figsize=(12, 6), xscale:str="linear", xscale2:str="linear", param1:str=None, param2:str=None):
    """Plot model cross-validation error along grid of hyperparameters

    Args:
        model: model to analyze
        figsize (tuple[float, float], optional): figure of plot size. Defaults to (12, 6).
        xscale (str, optional): Scale of x-axis of first plot. Defaults to "linear".
        xscale2 (str, optional): Scale of x-axis of second plot. Defaults to "linear".
        param1 (str, optional): First parameter of the grid to analyze. Defaults to None.
        param2 (str, optional): Second parameter of the grid to analyze. Defaults to None.

    Raises:
        TypeError: No hyperparameters found in grid, grid must have some hyperparameter to create plot
    """
    cv_r = model.cv_results_
    err = cv_r["mean_test_score"]
    std = cv_r["std_test_score"]
    param_names = list(model.cv_results_.keys())
    if param1 is not None and param2 is not None:
        param_names = ["param_"+param1, "param_"+param2]
    param_keys = [s for s in param_names if "param_" in s]
    params = [s.split("param_")[1] for s in param_keys]
        
    best_params = model.best_params_
    if not param_keys:
        raise TypeError("No hyperparameters encountered in grid.")
    if len(param_keys) > 1:
        grid1 = model.cv_results_[param_keys[0]].data
        cat1 = 'num'
        if not(type(grid1[0]) == int or type(grid1[0]) == float):
            grid1 = [p for p in list(grid1)]
            cat1 = 'cat'
        param_name1 = " ".join(params[0].split("__")[1].split("_"))
        grid2 = model.cv_results_[param_keys[1]].data
        cat2 = 'num'
        if not(type(grid2[0]) == int or type(grid2[0]) == float):
            grid2 = [p  for p in list(grid2)]
            cat2 = 'cat'
        param_name2 = " ".join(params[1].split("__")[1].split("_"))

        cols        = ['cv_error', 'cv_std']
        multi_index = pd.MultiIndex.from_tuples([(p1, p2) for p1, p2 in sorted(zip(grid1, grid2))], names=[param1, param2])
        dfe         = pd.DataFrame([(e, s) for e, s in zip(err, std)], columns=cols, index=multi_index)
        # First hyperparameter
        plt.figure(figsize=figsize)
        ax = plt.gca()
        dfe.unstack(level=1)['cv_error'].plot(ax=ax, style='o-', yerr=dfe.unstack(level=1)['cv_std'])
        #reset color cycle so that the marker colors match
        ax.set_prop_cycle(None)
        #plot the markers
        sc = dfe.unstack(level=1)['cv_error'].plot(figsize=(12,8), style='o-', markersize=5, ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        handles = handles[0:int(len(labels)/2)]
        labels = labels[0:int(len(labels)/2)]
        ax.legend(handles,labels,loc="lower right", title=param_name2)
        if not cat1 == 'cat':
            plt.plot(model.best_params_[params[0]], model.best_score_, marker="o", markersize=15, color="red")
        else:
            pos = list(dfe.unstack(level=1).index).index(model.best_params_[params[0]])
            plt.plot(pos, model.best_score_, marker="o", markersize=15, color="red")
        plt.title(f"Best model with {params[0]} = {str(best_params[params[0]])} and {params[1]} = {str(best_params[params[1]])} ")
        plt.xlabel(param_name1)
        plt.xscale(xscale)
        plt.show()
        # Second hyperparameter
        plt.figure(figsize=figsize)
        dfe.unstack(level=0)['cv_error'].plot(ax=plt.gca(), style='o-', yerr=dfe.unstack(level=1)['cv_std'])
        #reset color cycle so that the marker colors match
        plt.gca().set_prop_cycle(None)
        #plot the markers
        dfe.unstack(level=0)['cv_error'].plot(figsize=(12,8), style='o-', markersize=5, ax = plt.gca())
        handles, labels = plt.gca().get_legend_handles_labels()
        handles = handles[0:int(len(labels)/2)]
        labels = labels[0:int(len(labels)/2)]
        plt.gca().legend(handles,labels,loc="lower right", title=param_name1)
        if not cat2 == 'cat':
            plt.plot(model.best_params_[params[1]], model.best_score_, marker="o", markersize=15, color="red")
        else:
            pos = list(dfe.unstack(level=0).index).index(model.best_params_[params[1]])
            plt.plot(pos, model.best_score_, marker="o", markersize=15, color="red")
        plt.title(f"Best model with {params[0]} = {str(best_params[params[0]])} and {params[1]} = {str(best_params[params[1]])} ")
        plt.xlabel(param_name2)
        plt.xscale(xscale2)
        plt.show()
    else:
        grid=model.cv_results_[param_keys[0]].data
        if not(type(grid[0]) == int or type(grid[0]) == float):
            grid = [p for p in list(grid)]
        param_name= " ".join(params[0].split("__")[1].split("_"))
        
        plt.figure(figsize=figsize)
        plt.errorbar(grid, err, yerr=std, linestyle="None", ecolor='lightblue')
        plt.plot(grid, err, marker="o", markersize=10, c='lightblue')
        plt.plot(model.best_params_[params[0]], model.best_score_, marker="o", markersize=15, color="red")
        plt.title(f"Best model with {params[0]} = {str(best_params[params[0]])} ")
        plt.xlabel(param_name)
        plt.xscale(xscale)
        plt.show()
    return

def dotplot(scores:dict, metric:str):
    """Create boxplot plots of cross-validation error of several models

    Args:
        scores (dict): dictionary containing CV error of each model
        metric (str): x-axis label
    """
    plt.xlabel(metric)
    plt.ylabel('')
    plt.title("Scores")
    scores_list = [score for key, score in scores.items()]
    for i in range(len(scores_list)):
        plt.boxplot(scores_list[i], positions=[i], vert=False)
    plt.yticks(list(range(len(scores_list))), list(scores.keys()))
    plt.show()
    
def PlotDataframe(df:pd.core.frame.DataFrame, output_name:str, factor_levels:int=7, figsize=(12, 6), bins:int=30):
    """Create plots of dataframe analogous to diagonal plots of seaborn pairplot

    Args:
        df (pd.core.frame.DataFrame): Tidy (long-form) dataframe where each column is a variable and each row is an observation
        output_name (str): name of output variable column
        factor_levels (int, optional): threshold of number of unique differenting numeric and categorical variables. Defaults to 7.
        figsize (tuple, optional): size of plot figure. Defaults to (12, 6).
        bins (int, optional): bins used in histogram plots. Defaults to 30.
    """
    nplots = df.shape[1]
    out_num = np.where(df.columns.values == output_name)[0]
    out_factor = False
    # Check if the output is categorical
    if df[output_name].dtype.name == 'category' or len(df[output_name].unique()) <= factor_levels:
        out_factor = True
        df[output_name] = df[output_name].astype('category')

    # Create subplots
    fig, axs = plt.subplots(math.floor(math.sqrt(nplots))+1, math.ceil(math.sqrt(nplots)), figsize=figsize)
    fig.tight_layout(pad=4.0)
    
    if out_factor:
        input_num = 0
        for ax in axs.ravel():
            if input_num < nplots:
                # Create plots
                if input_num == out_num:
                    df.groupby(output_name).size().plot.bar(ax=ax, rot=0)
                    ax.set_title('Histogram of ' + output_name)
                else:
                    if df.iloc[:,input_num].dtype.name == 'category':
                        df.groupby([output_name,df.columns.values.tolist()[input_num]]).size().unstack().plot(kind='bar', ax=ax, rot=0)
                        ax.set_title(df.columns.values.tolist()[input_num] + ' vs ' + output_name)
                    else:
                        df.pivot(columns=output_name, values=df.columns.values.tolist()[input_num]).plot.hist(bins=bins, ax=ax,rot=0)
                        ax.set_title(df.columns.values.tolist()[input_num] + ' vs ' + output_name)

                input_num += 1
            else:
                ax.axis('off')

    else:
        input_num = 0
        for ax in axs.ravel():
            if input_num < nplots:
                # Create plots
                if input_num == out_num:
                    df[output_name].plot.hist(bins=bins,ax=ax)
                    ax.set_title('Histogram of ' + output_name)
                else:
                    if df.iloc[:,input_num].dtype.name == 'category':
                        sns.boxplot(x=df.columns.values.tolist()[input_num], y=output_name, data=df, ax=ax)
                        ax.set_title(df.columns.values.tolist()[input_num] + ' vs ' + output_name)
                    else:
                        sns.regplot(x=df.columns.values.tolist()[input_num], y=output_name, data=df, scatter_kws={'alpha': 0.5, 'color':'black'},line_kws={'color':'navy'}, ax=ax)
                        ax.set_title(df.columns.values.tolist()[input_num] + ' vs ' + output_name)

                input_num += 1
            else:
                ax.axis('off')

    # Plot the plots created
    plt.show()
    
#Formula transformer 
# https://juanitorduz.github.io/formula_transformer/
from patsy import dmatrices
from patsy import dmatrix
from sklearn.base import BaseEstimator, TransformerMixin
import patsy
class FormulaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, formula, feature_names=None):
        self.formula = formula
        self.feature_names = feature_names
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_formula = patsy.dmatrix(formula_like=self.formula, data=X)
        self.feature_names = X_formula.design_info.column_names
        return pd.DataFrame(X_formula, columns=self.feature_names)
    
    def get_feature_names_out(self):
        return np.array(self.feature_names)
    
#Custom Transformer that extracts columns passed as argument to its constructor 
class FeatureSelector( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__( self, feature_names ):
        self.feature_names = feature_names 
    
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        return X[ self.feature_names ]