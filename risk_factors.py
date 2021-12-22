# import libraries
import pandas as pd
import numpy as np
from scipy.spatial import distance

# high-low spread estimator (hlse)
@timethis
def hlse(ohlc_df, frequency='daily'):
    """
    Computes the high-low spread estimator, an estimate of bid-offer spreads, a measure of liquidity risk.
    See Corwin & Schultz (2011) for details: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1106193

    Parameters
    ----------
    ohlc_df: DataFrame
        DataFrame with DatetimeIndex and Open, High, Low and Close (OHLC) prices from which to compute the high-low spread estimates.
    frequency: str, {'daily', 'weekly', 'monthly'}, default 'daily'
        daily: daily bid-offer spread estimate.
        weekly: weekly bid-offer spread estimate, resampled over a weekly frequency as the mean of daily estimates.
        monthly: monthly bid-offer spread estimate, resampled over a monthly frequency as the mean of daily estimates.

    Returns
    -------
    S: Series
        Datetimeindex and time series of high-low spread estimates.
    """

    # define vars: mid, 2 day high and 2 day low vars
    mid, high_2d, low_2d = (ohlc_df.high + ohlc_df.low)/2, ohlc_df.high.rolling(2).max(), ohlc_df.low.rolling(2).min()

    # compute adjustment for overnight price moves
    ohlc_df['gap_up'], ohlc_df['gap_down'] = ohlc_df.low - ohlc_df.close.shift(1), ohlc_df.high - ohlc_df.close.shift(1)
    # adjustment for gap up
    ohlc_df['high_adj'], ohlc_df['low_adj'] = np.where(ohlc_df.gap_up > 0, ohlc_df.high - ohlc_df.gap_up, ohlc_df.high), np.where(ohlc_df.gap_up > 0, ohlc_df.low - ohlc_df.gap_up, ohlc_df.low)
    # adjustment for gap down
    ohlc_df['high_adj'], ohlc_df['low_adj'] = np.where(ohlc_df.gap_down < 0, ohlc_df.high - ohlc_df.gap_down, ohlc_df.high), np.where(ohlc_df.gap_down < 0, ohlc_df.low - ohlc_df.gap_down, ohlc_df.low)

    # B beta
    B = (np.log(ohlc_df.high_adj/ohlc_df.low_adj))**2 + (np.log(ohlc_df.high_adj.shift(1)/ohlc_df.low_adj.shift(1)))**2
    # G gamma
    G = (np.log(high_2d/low_2d))**2
    # alpha
    alpha = ((np.sqrt(2 * B) - np.sqrt(B)) / (3 - 2 * np.sqrt(2))) - (np.sqrt(G/(3 - 2 * np.sqrt(2))))
    # replace negative values by 0
    alpha = pd.Series(np.where(alpha < 0, 0, alpha), index=alpha.index)
    # substitute alpha into equation 14 to get high-low spread estimate S
    S = (2 * (np.exp(alpha) - 1)) / (1 + np.exp(alpha))
    # resample using daily mean
    if frequency == 'weekly':
        S = S.resample('W').mean()
    if frequency == 'monthly':
        S = S.resample('M').mean()
    # drop NaNs
    S.dropna(inplace=True)

    return S

# turbulence index
@timethis
def turbulence(ret_df, window_type='rolling', lookback=36, p_vals=False):
    """
    Computes the Mahalanobis distance from a basket of assset returns, aka the turbulence index.
    Turbulence measures the statistical unusualness of a set of returns given their historical pattern of behavior.
    High turbulence occurs when both the correlation and volatility of a basket of returns is far from the norm.
    As such, it can be a useful measure of tail risk at a portfolio level, i.e. when diversification is most likely to fail.
    For more details, see Skulls, Financial Turbulence, and Risk Management by Kritzman and Li (2010):
    https://www.tandfonline.com/doi/abs/10.2469/faj.v66.n5.3

    Parameters
    ----------
    ret_df: DataFrame
        DataFrame with DatetimeIndex and returns.
    window_type: str, {'fixed', 'expanding', 'rolling'}, default 'fixed'
        Provide a window type. If None, all observations are used in the calculation.
    lookback: int
        Number of observations to include in the window. If 'fixed' window_type, all observations will be used.
    p_vals: bool (optional), default False
        Provides the p-values of each turbulence observation. Those with p-values < 0.001 are generally considered outliers.

    Returns
    -------
    df: DataFrame
        DataFrame with DatetimeIndex and turbulence index series
    """
    # drop NaNs
    ret_df.dropna(inplace=True)

    # create emtpy df
    df = pd.DataFrame()
    # set window size
    window_size = lookback

    # if window type fixed
    if window_type =='fixed' or window_type is None:
        # demean returns
        ret_diff = ret_df.subtract(ret_df.mean()).to_numpy()
        # compute inverse covariance matrix
        inv_cov = np.linalg.pinv(ret_df.cov())
        # compute mahalanobis distance
        md = (ret_diff @ inv_cov @ ret_diff.T).diagonal()
        # create df
        df = pd.DataFrame(md, index=ret_df.index, columns=['turb'])

    else:
        # while loop for expanding or rolling window
        while window_size <= len(ret_df):
            # if window type expanding
            if window_type =='expanding':
                ret = ret_df.iloc[:window_size]
            # if window type rolling
            if window_type == 'rolling':
                ret = ret_df.iloc[window_size - lookback:window_size]
            # demean returns
            ret_diff = ret.subtract(ret.mean()).to_numpy()
            # compute inverse covariance matrix
            inv_cov = np.linalg.pinv(ret.cov())
            # compute mahalanobis distance
            md = (ret_diff @ inv_cov @ ret_diff.T).diagonal()
            # create df
            md_df = pd.DataFrame(md, index=ret.index, columns=['turb'])
            df = pd.concat([df, md_df.iloc[-1].to_frame().T])
            window_size += 1

    # include p_vals
    if p_vals == True:
        #calculate p-val for each turb val
        df['turb_pval'] = 1 - stats.chi2.cdf(df.turb, ret_df.shape[1] - 1)

    return df
