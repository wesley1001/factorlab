# import libraries
import pandas as pd
import numpy as np
from scipy import stats

# compute price log returns
def returns(series, lags=1):
    """
    Computes the log returns of a price series.

    Parameters
    ----------
    series: Series or DataFrame
        DatetimeIndex and price series.
    lags: int, default 1
        Interval over which to compute returns: 1 computes daily return, 5 weekly returns, etc

    Returns
    -------
    ret: Series or DataFrame
        Log differenced returns of price series.
    """
    # compute log returns
    ret = np.log(series) - np.log(series).shift(lags)
    # drop emtpy rows
    ret.dropna(how='all', inplace=True)

    return ret

# set target volatility for macro factors
def target_vol(ret_df, ann_vol=0.15):
    """
    Set volatility of returns to be equal to a specific vol target.

    Parameters
    ----------
    ret_df: DataFrame
        DataFrame with DatetimeIndex and returns to be vol-adjusted.
    ann_vol: float
        Target annualized volatility.

    Returns
    -------
    df: DataFrame
        DataFrame with vol-adjusted returns.
    """

    # set target vol
    norm_factor = 1 / ((ret_df.std() / ann_vol) * np.sqrt(12))
    df = ret_df * norm_factor

    return df

# data normalization of a time series using various methods
@timethis
def normalize(features, window_type='fixed', lookback=10, method='z-score'):

    """
    Normalizes features using the method selected.

    Parameters
    ----------
    features: Series or DataFrame
        DatetimeIndex and features to normalize.
    window_type: str, {'fixed', 'expanding', 'rolling}, default 'fixed'
        Provide a window type. If None, all observations are used in the calculation.
    lookback: int, default 10
        Size of the moving window. This is the minimum number of observations used for the rolling or expanding statistic.
    method: str, {'z-score', 'quantile', 'min-max', 'percentile'}, default 'z-score'
            z-score: subtracts mean and divides by standard deviation.
            quantile:  subtracts median and divides by interquartile range.
            min-max: brings all values into the range [0,1] by subtracting the min and divising by the range (max - min).
            percentile: converts values to their percentile rank relative to the observations in the defined window type.

    Returns
    -------
    norm_features: Series or DataFrame
        DatetimeIndex and normalized features
    """

    # rolling window type
    if window_type == 'rolling':
        # z-score method
        if method == 'z-score':
            norm_features = (features - features.rolling(lookback).mean()) / features.rolling(lookback).std()
        # quantile method
        elif method == 'quantile':
            norm_features = (features - features.rolling(lookback).median()) / (features.rolling(lookback).quantile(0.75) - features.rolling(looback).quantile(0.25))
        # min-max method
        elif method == 'min-max':
            norm_features = (features - features.rolling(lookback).min()) / (features.rolling(lookback).max() - features.rolling(lookback).min())
        # percentile method
        elif method == 'percentile':
            norm_features = features.rolling(lookback).apply(lambda x: stats.percentileofscore(x, x[-1])/100, raw=True)
        # None
        else:
            norm_features = features

    # expanding window type
    elif window_type == 'expanding':
        # z-score method
        if method == 'z-score':
            norm_features = (features - features.expanding(lookback).mean()) / features.expanding(lookback).std()
        # quantile method
        elif method == 'quantile':
            norm_features = (features - features.expanding(lookback).median()) / (features.expanding(lookback).quantile(0.75) - features.expanding(looback).quantile(0.25))
        # min-max method
        elif method == 'min-max':
            norm_features = (features - features.expanding(lookback).min()) / (features.expanding(lookback).max() - features.expanding(lookback).min())
        # percentile method
        elif method == 'percentile':
            norm_features = features.expanding(lookback).apply(lambda x: stats.percentileofscore(x, x[-1])/100, raw=True)
        # None
        else:
            norm_features = features

    # fixed window type
    else:
        # z-score method
        if method == 'z-score':
            norm_features = (features - features.mean()) / features.std()
        # quantile method
        elif method == 'quantile':
            norm_features = (features - features.median()) / (features.quantile(0.75) - features.quantile(0.25))
        # min-max method
        elif method == 'min-max':
            norm_features = (features - features.min()) / (features.max() - features.min())
        # percentile method
        elif method == 'percentile':
            norm_features = features.rank(pct=True)
        # None
        else:
            norm_features = features

    # drop NaNs
    norm_features = norm_features.dropna(how='all')

    return norm_features
