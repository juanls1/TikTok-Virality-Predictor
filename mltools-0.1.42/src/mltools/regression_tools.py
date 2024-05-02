from datetime import datetime
from numpy.linalg import matrix_rank, svd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as mtrs
import statsmodels.api as sm
from scipy import stats

plt.style.use('ggplot')



def summaryLinReg(model, X: pd.core.frame.DataFrame, y: pd.core.series.Series, use_old:bool= True):
    """Summary of scikit 'LinearRegression' models.
    
    Provide feature information of linear regression models,
    such as coefficient, standard error and p-value. It is adapted
    to stand-alone and Pipeline scikit models.
    
    Important restriction of the function is that LinearRegression 
    must be the last step of the Pipeline.

    Args:
        model: LinearRegression or Pipeline model
        X (pd.core.frame.DataFrame): Input variables dataframe
        y (pd.core.series.Series): Output variable series
        use_old (bool): use previous version of summary of linear regression, useful when multicollinearity breaks new method. Default to True.
    """
    if use_old:
        # Obtain coefficients of the model
        if type(model) is LinearRegression:
            coefs = model.coef_
            intercept = model.intercept_
        else:
            coefs = model[len(model) - 1].coef_ #We suppose last position of pipeline is the linear regression model
            intercept = model[len(model) - 1].intercept_
        
        if type(model) is Pipeline:
            X_design = model[0].transform(X)
            coefnames = list()
            if hasattr(model[0],"transformers_"):
                for tt in range(len(model[0].transformers_)):
                    try:
                        if hasattr(model[0].transformers_[tt][1].steps[-1][1],"get_feature_names"):
                            aux = model[0].transformers_[tt][1].steps[-1][1].get_feature_names_out(model[0].transformers_[tt][2])
                            if type(aux)==list:
                                coefnames = coefnames + aux
                            else:
                                coefnames = coefnames + aux.tolist()
                        else:
                            coefnames = coefnames + model[0].transformers_[tt][2]
                    except:
                        continue
            else:
                coefnames = X.columns.values.tolist()
        
            
        ## include constant ------------- 
        if model[len(model) - 1].intercept_ != 0:
            coefnames.insert(0,'Intercept')
            if type(X_design).__module__ == np.__name__:
                X_design = np.hstack([np.ones((X_design.shape[0], 1)), X_design])
                X_design = pd.DataFrame(X_design, columns=coefnames)
            elif type(X_design).__module__ == 'pandas.core.frame':
                pass
            else:
                X_design = np.hstack([np.ones((X_design.shape[0], 1)), X_design.toarray()])
                X_design = pd.DataFrame(X_design, columns=coefnames)    
        else:
            try:
                X_design = X_design.toarray()
                X_design = pd.DataFrame(X_design, columns=coefnames)
            except:
                pass
        

        ols = sm.OLS(y.values, X_design)
        ols_result = ols.fit()
        return ols_result.summary()
    else:
        # Obtain coefficients of the model
        if type(model) is LinearRegression:
            coefs = model.coef_
            intercept = model.intercept_
            input_names = X.columns
            X_t = X
        else:
            coefs = model[len(model) - 1].coef_ #We suppose last position of pipeline is the linear regression model
            intercept = model[len(model) - 1].intercept_
            X_t = model[len(model) - 2].transform(X)
            input_names = model[len(model) - 2].get_feature_names_out().tolist()

        intercept_included = True
        params = coefs
        if not intercept == 0 and not 'Intercept' in input_names:
            params = np.append(intercept,coefs)
            input_names = ['Intercept'] + input_names
            intercept_included = False
        elif 'Intercept' in input_names:
            coefs[input_names.index('Intercept')] = intercept
            params = coefs
        predictions = model.predict(X)
        residuals = y - predictions

        print('Residuals:')
        quantiles = np.quantile(residuals, [0,0.25,0.5,0.75,1], axis=0)
        quantiles = pd.DataFrame(quantiles, index=['Min','1Q','Median','3Q','Max'])
        print(quantiles.transpose())
        # Note if you don't want to use a DataFrame replace the two lines above with
        if not intercept_included:
            if not type(X_t) == pd.core.frame.DataFrame and not type(X_t) == np.ndarray:
                newX = np.append(np.ones((len(X_t.toarray()),1)), X_t.toarray(), axis=1)
            else:
                newX = np.append(np.ones((len(X_t),1)), X_t, axis=1)
        else:
            if not type(X_t) == pd.core.frame.DataFrame:
                newX = X_t.toarray()
            else:
                newX = X_t.values
            
        MSE = (sum((residuals)**2))/(len(newX)-len(newX[0]))

        var_b = MSE * (np.linalg.inv(np.dot(newX.T,newX)).diagonal())
        sd_b = np.sqrt(var_b)
        ts_b = params / sd_b

        p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-len(newX[0])))) for i in ts_b]

        sd_b = np.round(sd_b,3)
        ts_b = np.round(ts_b,3)
        p_values = np.round(p_values,3)
        params = np.round(params,4)

        myDF3 = pd.DataFrame()
        myDF3["Input"], myDF3["Coefficients"], myDF3["Standard Errors"], myDF3["t values"], myDF3["Pr(>|t|)"], myDF3["Signif"] = [input_names, params, sd_b, ts_b, p_values, np.vectorize(lambda x: '***' if x < 0.001 else ('**' if x < 0.01 else ('*' if x < 0.05 else ('.' if x < 0.1 else ' '))))(p_values)]
        myDF3.set_index("Input", inplace=True)
        myDF3.index.name = None
        print(myDF3)
        print('---\nSignif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1\n')
        print(f'Residual Standard Error: {round(np.std(residuals), 3)} on {residuals.shape[0]} degrees of freedom')
        # error metrics
        r2 = mtrs.r2_score(y, predictions)
        n = len(y)
        k = X.shape[1]
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        RMSE = np.sqrt(mtrs.mean_squared_error(y, predictions))
        MAE = mtrs.mean_absolute_error(y, predictions)
        print(f'R-squared: {round(r2, 3)}, Adjusted R-squared: {round(adjusted_r2, 3)}, RMSE: {round(RMSE, 3)}, MAE: {round(MAE, 3)}')
        # F test
        TSS = np.sum(y - predictions) ** 2
        RSS = (1 - r2) * TSS
        num_f = (TSS - RSS) / k
        den_f = RSS / (n - k - 1)
        f = num_f / den_f
        p_f = 1 - stats.f.cdf(f, k, (n - k - 1)) 
        print(f'F-statistic: {round(f, 3)} on {k} and {n - k - 1} DOF, P(F-statistic): {round(p_f, 3)}')
        return 

def ObtainCoefnamesPipe(model,X_train):
    #obtain coefnames
    coefnames = list()
    if hasattr(model[0],"transformers_"):
        for tt in range(len(model[0].transformers_)):
            if hasattr(model[0].transformers_[tt][1].steps[-1][1],"get_feature_names"):
                aux = model[0].transformers_[tt][1].steps[-1][1].get_feature_names(model[0].transformers_[tt][2])
                if type(aux)==list:
                    coefnames = coefnames + aux
                else:
                    coefnames = coefnames + aux.tolist()
            else:
                coefnames = coefnames + model[0].transformers_[tt][2]
    else:
        coefnames = X_train.columns.values.tolist()
    return coefnames


def plotModelDiagnosis(df, pred, output_name, figsize=(12,6), bins=30, smooth_order=5):
    """
    Plot model diagnosis for regression analysis.
    
    This function generates diagnostic plots for regression analysis, including
    residual histograms and scatter plots or boxplots of residuals against each
    variable in the dataframe. For numerical variables, a scatter plot of residuals
    against the variable is created with a smoothed line. For categorical variables,
    a boxplot of residuals against categories is created.
    
    Parameters
    ----------
    df : pd.core.frame.DataFrame
        Dataframe containing input, output, and prediction variables.
        
    pred : str
        Name of the column in `df` containing the model predictions.
        
    output_name : str
        Name of the column in `df` containing the actual output values.
        
    figsize : tuple of (float, float), optional, default=(12, 6)
        Width and height of the figure in inches.
        
    bins : int, optional, default=30
        Number of bins to use in the histogram of residuals.
        
    smooth_order : int, optional, default=5
        Degree of the smoothing spline in the scatter plots of residuals against
        numerical variables.
    
    Returns
    -------
    None
        The function creates plots using Matplotlib and does not return any value.
    
    Examples
    --------
    >>> plotModelDiagnosis(df, 'predictions', 'actual_output')
    """
    # Create the residuals
    df['residuals'] = df[output_name] - df[pred]
    out_num = np.where(df.columns.values == 'residuals')[0]
    nplots = df.shape[1]
    
    # Create subplots
    fig, axs = plt.subplots(
        math.floor(math.sqrt(nplots)) + 1, 
        math.ceil(math.sqrt(nplots)), 
        figsize=figsize
    )
    fig.tight_layout(pad=4.0)

    input_num = 0
    for ax in axs.ravel():
        if input_num < nplots:
            # Create plots
            if input_num == out_num:
                df['residuals'].plot.hist(bins=bins, ax=ax)
                ax.set_title('Histogram of residuals')
            else:
                if df.iloc[:,input_num].dtype.name == 'category':
                    sns.boxplot(x=df.columns.values.tolist()[input_num], y='residuals', data=df, ax=ax)
                    ax.set_title(df.columns.values.tolist()[input_num] + ' vs ' + 'residuals')
                else:
                    sns.regplot(
                        x=df.columns.values.tolist()[input_num], 
                        y='residuals', 
                        data=df, 
                        ax=ax, 
                        order=smooth_order, 
                        ci=None, 
                        line_kws={'color':'navy'}
                    )
                    ax.set_title(df.columns.values.tolist()[input_num] + ' vs ' + 'residuals')

            input_num += 1
        else:
            ax.axis('off')

class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    LinearRegressor is a wrapper for using statsmodels' OLS with scikit-learn.

    This class allows the use of statsmodels' OLS model with scikit-learn's
    GridSearchCV by providing `fit` and `predict` methods, and implementing
    `get_params` and `set_params` methods.

    Parameters
    ----------
    fit_intercept : bool, optional (default=True)
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations.

    Attributes
    ----------
    coef_ : array of shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.

    intercept_ : float or array of shape (n_targets,)
        Independent term in the linear model. Set to 0.0 if fit_intercept = False.

    rank_ : int
        Rank of matrix X. Only available when X is dense.

    singular_ : array of shape (min(X, y),)
        Singular values of X. Only available when X is dense.

    n_features_in_ : int
        Number of features seen during fit.
    
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit. Defined only when X has feature names
        that are all strings.
    """
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.rank_ = None
        self.singular_ = None
        self.intercept_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        
    def fit(self, X, y):
        """
        Fit the OLS model according to the given training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target vector relative to X.

        Returns
        -------
        self : object
            Returns self.
        """
        
        if self.fit_intercept:
            X = np.column_stack((np.ones((X.shape[0], 1)), X.toarray() if hasattr(X, 'toarray') else X))
        else:
            X = X.toarray() if hasattr(X, 'toarray') else X
        
        self.n_features_in_ = X.shape[1]
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.to_numpy()
        else:
            self.feature_names_in_ = np.arange(self.n_features_in_)
        
        self.rank_ = matrix_rank(X)
        _, s, _ = svd(X, full_matrices=False)
        self.singular_ = s
        
        self.model_ = sm.OLS(y, X)
        self.results_ = self.model_.fit()
        
        if self.fit_intercept:
            self.intercept_ = self.results_.params.iloc[0]
            self.coef_ = self.results_.params.iloc[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = self.results_.params
        
        return self
    
    def predict(self, X):
        """
        Predict using the linear model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape (n_samples,) or (n_samples, n_outputs)
            Returns predicted values.
        """
        if self.fit_intercept:
            X = np.column_stack((np.ones((X.shape[0], 1)), X.toarray() if hasattr(X, 'toarray') else X))
        else:
            X = X.toarray() if hasattr(X, 'toarray') else X
        return self.results_.predict(X)
    
    def summary(self, input_names=None):
        """
        Print a summary of the regression self.results_.

        This method generates a summary of the regression model results, 
        which includes various statistical metrics, such as coefficients 
        of the independent variables, R-squared value, hypothesis test 
        results, etc. Optionally, it allows you to specify custom names 
        for the independent variables for a more readable summary.

        Parameters
        ----------
        input_names : list of str, optional
            A list of names to be used as the names of the independent variables 
            in the model summary. This list should be of the same length as 
            the number of independent variables and in the same order. 
            If None (default), the existing names in the model are used.

        Examples
        --------
        >>> model = LinearRegressor()
        >>> model.fit(X, y)
        >>> model.summary()
        >>> model.summary(input_names=['var1', 'var2', 'var3'])
        """
        if input_names is None:
            input_names = self.results_.model.exog_names
        elif not isinstance(input_names, list):
            input_names = list(input_names)
            
        if self.fit_intercept:
            input_names = ['const'] + input_names
        # Print basic model info and fit statistics
        print(self.results_.summary().tables[0])
        
        # Print coefficient estimates
        print("{:<25} {:<15} {:<15} {:<15} {:<15}".format("var", "coef", "std err", "t", "P>|t|"))
        print("-" * 75)
        for name, coef, stderr, tvalue, pvalue in zip(input_names, self.results_.params, self.results_.bse, self.results_.tvalues, self.results_.pvalues):
            print("{:<25} {:<15.3f} {:<15.3f} {:<15.3f} {:<15.3f}".format(name, coef, stderr, tvalue, pvalue))
        
        print(self.results_.summary().tables[2])
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {"fit_intercept": self.fit_intercept}
    
    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self