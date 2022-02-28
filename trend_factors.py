# import libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm

### Trend Factors
# breakout
def breakout(series, lookback=14):
    """
    Compute breakout trend factor signal.

    Parameters
    ----------
    series: Series or Dataframe
        Series or DataFrame with DatetimeIndex and price series.
    lookback: int
        Number of observations to include in the rolling window.

    Returns
    -------
    signal: Series or DataFrame
        Series or DataFrame with DatetimeIndex and breakout signal.
    """

    # compute rolling looback high and low
    high = series.rolling(lookback).max().shift(1)
    low = series.rolling(lookback).min().shift(1)

    # compute signal
    signal = series.copy()
    signal.iloc[:] = 0
    signal[series > high] = 1
    signal[series < low] = -1

    # ffill 0s to create always invested strategy
    signal.replace(to_replace=0, method='ffill', inplace=True)

    return signal


# moving window difference
def mw_diff(series, log=True, short=3, long=14, lag=0, central_tendency='mean'):
    """
    Computes the moving window difference trend factor.

    Parameters
    ----------
    series: Series or Dataframe
        Series or DataFrame with DatetimeIndex and price series.
    log: bool, default False
        Computes log of price series.
    short: int, default 5
        Number of observations to include in the short rolling window.
    long: int, default 20
        Number of observations to include in the long rolling window.
    lag: int, default 0
        Number of observations to lag the long rolling window.
    central_tendency: str, {'mean', 'median'}, default 'mean'
        Measure of central tendency to use for smoothing over rolling window.


    Returns
    -------
    mw_diff: Series or DataFrame
        Series or DataFrame with DatetimeIndex and moving window difference trend factor.
    """

    # log parameter
    if log is True:
        series = np.log(series)

    # median
    if central_tendency == 'quantile':
        mw_diff = series.rolling(short).median() - series.rolling(long).median().shift(lag)
    # mean or none
    else:
        mw_diff = series.rolling(short).mean() - series.rolling(long).mean().shift(lag)

    return mw_diff


# price momentum
def price_mom(series, log=True, lookback=14):
    """
    Computes the price momentum trend factor.

    Parameters
    ----------

    series: Series or Dataframe
        Series or DataFrame with DatetimeIndex and price series.
    log: bool, default True
        Computes log of price series.
    lookback: int
        Number of observations to include in the rolling window.

    Returns
    -------
    price_mom: Series or DataFrame
       Price change over n-day lookback window.
    """

    # log parameter
    if log is True:
        price_mom = np.log(series) - np.log(series).shift(lookback)  # log of price series
    else:
        price_mom = series - series.shift(lookback)  # price series

    return price_mom


# price velocity, price acceleration and price jerk
def price_dynamics(series, log=True, lookback=14, coef=1):
    """
    Compute the price dynamics indicator by regressing price on a time trend to estimate coefficients.

    Parameters
    ----------
    df: Series or DataFrame
        Series or DataFrame with DatetimeIndex and price series.
    log: bool, default True
        Computes log of price series.
    lookback: int
        Number of observations to include in the rolling window regression.
    coef: int, {0, 1, 2, 3}, default 1
        coefficient estimate, 0 for intercept, 1 for velocity, 2 for acceleration and 3 for jerk

    Returns
    -------
    coeff: Series
        Coefficients of price regressed on a time trend over n-day lookback window
    """

    # convert Series to DataFrame
    if isinstance(series, pd.Series):
        series = series.to_frame()

    # log parameter
    if log:
        series = np.log(series)  # log of price series

    # create empty df to store coeff series
    df1 = pd.DataFrame()

    # loop through df
    for col in series.columns:

        # set rolling window size
        window = lookback

        # create empty df to store coeff values
        df2 = pd.DataFrame(index=series.index, columns=[col])

        # keep going until window window size reaches full sample window
        while window <= series.shape[0]:
            # regression
            y = series[col].iloc[window - lookback:window]
            X = sm.tsa.tsatools.add_trend(y, trend='ct', prepend=False).loc[:, 'const':]
            X['t2'] = X.trend ** 2
            X['t3'] = X.trend ** 3

            # Fit and summarize OLS model
            mod = sm.OLS(y, X)
            res = mod.fit()

            # add beta to df
            df2.loc[y.index[-1], col] = res.params[coef]

            # roll window forward one day
            window += 1

            # add col to df and rename col by ticker
        df1 = pd.concat([df1, df2], axis=1)

    return df1


# RSI
def rsi(series, lookback=14, smoothing=None):
    """
    Computes the RSI indicator.

    Parameters
    ----------
    series: Series or Dataframe
        Series or DataFrame with DatetimeIndex and price series.
    lookback: int
        Number of observations to include in the rolling window for RSI.
    smoothing: str, {'sma', 'ewma'}, defaul 'sma'
        sma: simple moving average;
        ema: for exponential moving average.

    Returns
    -------
    rsi: series or DataFrame
        RSI indicator.
    """

    # compute price returns and up/down days
    ret = np.log(series) - np.log(series).shift(1)
    up = ret.where(ret > 0).fillna(0)
    down = ret.where(ret < 0).fillna(0)

    # smoothing parameter
    if smoothing == 'ema':
        rs = up.ewm(span=lookback, min_periods=1).mean() / (down.ewm(span=lookback, min_periods=1).mean() * -1)
    else:
        rs = up.rolling(lookback, min_periods=lookback).mean() / (
                    down.rolling(lookback, min_periods=lookback).mean() * -1)

    rsi = 100 - (100 / (1 + rs))

    return rsi


# stochastic
def stochastic(ohlc_df, k_lookback=14, d_lookback=3, smoothing=None):
    """
    Computes the stochastic indicator K and D.

    Parameters
    ----------
    ohlc_df: DataFrame
        DataFrame with DatetimeIndex and OHLC prices.
    k_lookback: int
        Number of observations to include in the rolling window for k stochastic calculation.
    d_lookback: int
        Number of observations to include in the rolling window for d stochastic calculation.
    smoothing: str, {'sma', 'ewma'}, default 'sma'
        'sma' simple moving average;
        ewma: exponential moving average .

    Returns
    -------
    stochastic k, d: DataFrame
        Stochastic k,d indicator.
    """

    # smoothing parameter
    if smoothing == 'ema':
        k = 100 * ((ohlc_df.close - ohlc_df.low.rolling(k_lookback).min()) / (
                    ohlc_df.high.rolling(k_lookback).max() - ohlc_df.low.rolling(k_lookback).min()))
        d = k.ewm(span=d_lookback).mean()
    else:
        k = 100 * ((ohlc_df.close - ohlc_df.low.rolling(k_lookback).min()) / (
                    ohlc_df.high.rolling(k_lookback).max() - ohlc_df.low.rolling(k_lookback).min()))
        d = k.rolling(d_lookback).mean()

    return pd.DataFrame({'stochastic_k': k, 'stochastic_d': d})


# intraday intensity (Aronson)
def intensity(ohlc_df, lookback=14, central_tendency='mean'):
    """
    Computes intraday intensity trend factor.

    Parameters
    ----------
    ohlc_df: DataFrame
        DataFrame with DatetimeIndex and OHLC prices.
    lookback: int
        Number of observations to include in the rolling window mean.
    central_tendency: str, {'mean', 'median'}, default 'mean'
        Measure of central tendency to use for smoothing over rolling window.

    Returns
    -------
    ii, cmf: DataFrame
        intensity index and Chaiken's money flow over n-day lookback window
    """

    # compute true range
    ohlc_df['high_low'], ohlc_df['high_close'], ohlc_df['low_close'] = ohlc_df.high - ohlc_df.close, abs(
        ohlc_df.high - ohlc_df.close.shift(1)), abs(ohlc_df.low - ohlc_df.close.shift(1))
    tr = ohlc_df.loc[:, 'high_low':].max(axis=1)
    # compute today's change
    today_chg = ohlc_df.close - ohlc_df.open
    # compute intensity
    intensity = today_chg / tr
    # central tendency of intensity over rolling window
    if central_tendency == 'median':
        intensity_smooth = intensity.rolling(lookback).median()
    else:
        intensity_smooth = intensity.rolling(lookback).mean()

    return intensity_smooth