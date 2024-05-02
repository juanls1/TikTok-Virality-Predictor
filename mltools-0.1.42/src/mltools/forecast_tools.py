import scipy.stats as st
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
def boxcox_lambda_plot(ts, window_width: int, height=10) -> float:
    """Create plot of log(mean) vs log(sd) of time series to check if Box-Cox transformation is needed to stabilize variance.

    If linear correlation of log(mean) and log(sd) is high (>=0.6), then there is a correlation between variance and level of
    time series, not assessing the necessary condition for ARIMA to have a constant variance through all the time series. 
    In that case, Box-Cox transformation is needed to stabilize variance of the timeseries. 
    Args:
        ts (pd.core.frame.DataFrame | pd.core.series.Series): Time series to be analyzed
        window_width (int): Number of samples of the window to calculate mean and standard deviation.
        height (int): height of the figure.
        
    Raises:
        ValueError: window_width must be at least 1 to calculate mean and std
        ValueError: window_width must be less or equal length of ts

    Returns:
        float: lambda value of box-cox transformation
    """
    # Box-Cox Transformation
    if window_width < 1:
        raise ValueError('window_width must be at least 1')
    if window_width > ts.shape[0]:
        raise ValueError(f'window_width {window_width} is greater than the number of data samples provided in ts')
    fm = min(ts.values)
    if fm < 0:
        ts = pd.DataFrame(ts.values - min(ts.values) + 1, columns=ts.columns.values.tolist())
    mm  = np.zeros((ts.shape[0] - window_width + 1, ))
    sdm = np.zeros((ts.shape[0] - window_width + 1, ))
    for i in range(sdm.shape[0]):
        mm[i]  = np.log(np.mean(ts.values[i:(i + window_width)]))
        sdm[i] = np.log(np.std(ts.values[i:(i + window_width)]))
    dfm = pd.DataFrame({'log_ma': mm, 'log_sd': sdm})
    # dfm = pd.concat([ts.rolling(window_width).mean(), ts.rolling(window_width).std()], axis=1)
    # dfm.columns = ['log_ma','log_sd']
    # dfm = dfm.iloc[window_width+1:,:]
    # dfm = dfm.reset_index()
    lm = LinearRegression().fit(dfm['log_ma'].values.reshape(-1, 1), dfm['log_sd'].values)
    r_value = lm.score(dfm['log_ma'].values.reshape(-1, 1), dfm['log_sd'].values)
    slope = lm.coef_[0]
    lambd = 1 - slope
    sns.jointplot(x='log_ma', y='log_sd', data=dfm, kind='reg', marginal_kws=dict(bins=15), height=height,
                joint_kws={'line_kws':{'color':'cyan'}, 'scatter_kws':{'alpha': 0.5, 'edgecolor':'black'}})
    # plt.gca().text(0.5*dfm['log_ma'].mean(), 0, 'Lambda = ' + str(round(lambd,2)) + '. R squared = ' + str(round(r_value,2)) + ', window width = ' + str(round(window_width,2)))
    plt.suptitle('Lambda = ' + str(round(lambd,4)) + '    R squared = ' + str(round(r_value,4)) + '    Window width = ' + str(window_width), y=1, fontsize=10)
    plt.show()
    return lambd

def check_residuals(ts, lags:int=25, bins:int=30, figsize=(10, 5)):
    """Create plot to check if residuals are white noise.

    If residuals are white noise, histogram shall reflect a normal distribution, ACF and PACF shall not have
    any notable correlation, and no peak shall be seen in the time series plot.
    Args:
        ts (pd.core.frame.DataFrame | pd.core.series.Series): Time series with the residuals to be analyzed
        lags (int, optional): Number of lags to create ACF and PACF plots. Defaults to 25.
        bins (int, optional): Bins of the histogram of the residuals. Defaults to 30.
        figsize (tuple, optional): size of the figure. Defaults to (10, 15).
    """
    figure = plt.figure(figsize=figsize)
    grid = plt.GridSpec(2, 2, figure=figure)
    # Plot series as is
    plt.subplot(grid[0, 0])
    plt.plot(ts)
    # Plot histogram of residuals
    ax = plt.subplot(grid[0, 1])
    # ts.plot(kind='hist', ax=ax, bins=bins, density=True, edgecolor='black', color='gray', label='Data',legend=False)
    # ts.plot(kind='kde', ax=ax, label='PDF', color='red',legend=False)
    # ax = sns.distplot(ts, hist=True, kde=True, bins=bins, ax=ax, color = 'darkblue', 
    #             hist_kws={'edgecolor':'black', 'color':'gray'},
    #             label="hist",
    #             kde_kws={'linewidth': 4})
    ax = sns.histplot(ts, kde=True, bins=bins, ax=ax,
                    alpha=0.4, 
                    edgecolor = 'red',
                    label="hist")
    # calculate the pdf
    x0, x1 = ax.get_xlim()  # extract the endpoints for the x-axis
    _, y1 = ax.get_ylim()  # extract the endpoints for the x-axis
    x_pdf = np.linspace(x0, x1, 100)
    y_pdf = norm.pdf(x_pdf)
    y_pdf = y1 * y_pdf / np.max(y_pdf)
    ax.plot(x_pdf, y_pdf, label='N(0, 1)', color='purple')
    plt.legend()
    plt.ylabel('Probability')
    # Plot ACF
    ax = plt.subplot(grid[1, 0])
    sm.graphics.tsa.plot_acf(ts.values.squeeze(), lags=lags, ax=ax, zero=False)
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    # Plot PACF
    ax = plt.subplot(grid[1, 1])
    sm.graphics.tsa.plot_pacf(ts.values.squeeze(), lags=lags, ax=ax, zero=False)
    ax.set_xlabel('Lag')
    ax.set_ylabel('PACF')
    plt.tight_layout(pad=4.0)
    print("Ljung-Box test of residuals:")
    print(sm.stats.acorr_ljungbox(ts, lags=[lags], return_df=True))

def inverse_box_cox(ts, lmbda: float):
    """Inverse Box-cox transformation

    Args:
        ts (pd.core.frame.DataFrame | pd.core.series.Series): time series to perform inverse Box-cox
        lmbda (float): lambda use to perform the Box-cox transformation

    Returns:
        pd.core.frame.DataFrame | pd.core.series.Series: time series resulted from inverse Box-cox
    """
    if lmbda == 0:
        return(np.exp(ts))
    else:
        return(np.exp(np.log(lmbda*ts+1)/lmbda))
    
def ts_display(ts, lags: int=25, figsize=(10, 5)):
    """Plot timeseries plot, ACF and PACF plots 

    Args:
        ts (pd.core.frame.DataFrame | pd.core.series.Series): time series to be plotted
        lags (int, optional): Lags to be calculated in ACF y PACF. Defaults to 25:int.
        figsize (tuple, optional): size of the figure. Defaults to (10, 15).
    """
    figure = plt.figure(figsize=figsize)
    grid = plt.GridSpec(2, 2, figure=figure)
    # Plot series as is
    plt.subplot(grid[0, :])
    plt.tight_layout(pad=4.0)
    plt.plot(ts)
    # Plot ACF
    ax = plt.subplot(grid[1, 0])
    sm.graphics.tsa.plot_acf(ts.values.squeeze(), lags=lags, ax=ax, zero=False)
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    # Plot PACF
    ax = plt.subplot(grid[1, 1])
    sm.graphics.tsa.plot_pacf(ts.values.squeeze(), lags=lags, ax=ax, zero=False)
    ax.set_xlabel('Lag')
    ax.set_ylabel('PACF')