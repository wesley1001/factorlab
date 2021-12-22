"""
This function takes observable macro factors (surprises) as inputs and creates macro factor mimicking portfolios (MFMPs) as outputs. It uses
a novel machine-learning approach, the Principal Components Instrumental Variables FMP Estimator,
described by Jurczenko and Teiletche (2020) in Macro Factor-Micking Portfolios:
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3363598
The methodology addresses many of the common problems associated with macro factors and multifactor risk modeling and, as they show,
is superior to other common FMP approaches.
We describe the steps of the PCIV algorithm below, as well as in our Medium post:

For a more detailed explanation, see the link to the paper above.
Note that access to macroeconomic and base assets is required to estimate macro factor-mimicking portfolios. See the macro factors
description table in our Medium post for details on macro factors and base assets. We suggest using your
preferred data provider(s) with historical data going as far back as possible in order to capture various macroeconomic regimes and
improve estimates. Like the authors, we use 50 years of monthly data history.
Observable macro factors, fitted macro factors, and factor-mimicking portfolios (returns) are provided in the 'macro_fmp.csv' file.
"""

# import packages
import pandas as pd
import numpy as np
import statsmodels.api as sm

# import functions from FactorLab
from factorlab import feature_engineering, factor_analysis, time_series_analysis, risk_factors

### define functions
# Estimate fitted macro factors from orthogonalized macro factors using tPCA (F yhat)
def fitted_macro_factors(obs_factors_df, base_assets_ret_df):
    """
    Estimates fitted macro factors using 2SLS instrumental variable estimation approach. First, LASSO regression is used to
    select a subset of relevant base asset returns for each observable macro factor. Then, that macro factor is projected
    onto the first L principal components of the subset of selected base asset returns to obtain the fitted values (F-yhat)
    for the macro factors.

    Parameters
    ----------
    obs_factors_df: DataFrame
        DataFrame with DatetimeIndex and orthogonalized observable macro factors.
    base_assets_ret_df: DataFrame
        DataFrame with DatetimeIndex and base asset returns.

    Returns
    -------
    fitted_factors_df: DataFrame
        DataFrame wiht DatetimeIndex and fitted macro factors (F_yhat).
    """

    # create emtpy df to store fitted factor estimates
    fitted_factors_df = pd.DataFrame()

    # iterate through obs factors df
    for factor in obs_factors_df.columns:

        # run tPCA to extract L principal components from the subset of relevant features
        pcs_df = tpca(obs_factors_df.loc[:,factor], base_assets_ret_df)

        # project obs macro factor onto PCs
        fitted_factor_series = project_target(mf_df.loc[:, factor], pcs_df)

        # add fitted factor to empty fitted factors df
        fitted_factors_df = pd.concat([fitted_factors_df, fitted_factor_series.to_frame(name= factor + '_yhat')], join='outer', axis=1)

    return fitted_factors_df

# regress asset returns individually on the fitted macro factors (F yhat) to get macro beta estimates B (N x K) in eq 1
def factor_betas(base_assets_ret_df, factors_df, output_format='df'):
    """
    Estimates factor betas by regressing base asset returns individually on fitted macro factors.

    Parameters
    ----------
    base_assets_df: DataFrame, T x N
        DataFrame with DatetimeIndex and base asset returns.
    factors_df: DataFrame, T x F
        DataFrame with DatetimeIndex and fitted macro factors.
    output_format: str, {'array', 'df'}, default 'df'
        Specify output format, numpy.array or DataFrame.

    Returns
    -------
    betas: DataFrame, N x K, where N is the number of base assets and K the number of factors
        DataFrame with factor beta estimates (B in eqn 1) for each base asset, where N is the number of base assets
        and K is the number of factors.
    """

    # estimate factor exposures using factor exposures fcn
    factor_betas_dict = factor_exposures(base_assets_ret_df, factors_df)
    # store factor betas from dict into empty beta df
    betas_df = pd.DataFrame()
    for asset in factor_betas_dict:
        betas_df[asset] = factor_betas_dict[asset].loc['beta',:]
    # transpose df
    betas = betas_df.T
    # drop first and  last col
    betas = betas.iloc[:,1:-1]

    # if output format is numpy
    if output_format == 'array':
        betas = betas.to_numpy()
    else:
        betas = betas

    return betas

# Compute portfolio weights using factor beta estimates from tPCA
def portfolio_weights(factor_betas_df, output_format='df'):
    """
    Estimates macro factor-mimicking portfolio weights.

    Parameters
    ----------
    factor_betas_df: DataFrame, N x K
        DataFrame with factor beta estimates for each base asset.
    output_format: str, {'array', 'df'}, default 'df'
        Specify output format, numpy.array or DataFrame

    Returns
    -------
    W: DataFrame, N x K, where N is the number of base assets and K the number of factors
        DataFrame with portfolio weights.
    """

    # convert DataFrame to numpy for faster and easier matrix computation
    if isinstance(factor_betas_df, pd.DataFrame):
        B = factor_betas_df.to_numpy()
    # compute weights
    W = B @ np.linalg.pinv(B.T @ B)
    if output_format != 'array':
        W = pd.DataFrame(W, index=factor_betas_df.index, columns=factor_betas_df.columns)

    return W

# compute FMP returns with portfolio weights
def create_fmp(portfolio_weights_df, base_assets_ret_df):
    """
    Computes macro factor-mimicking portfolios from portfolio weights and base asset returns.

    Parameters
    ----------
    factor_weights_df: DataFrame, N x K
        DataFrame with portfolio weights for each base asset.
    base_assets_ret_df: DataFrame, T x N
        DataFrame with DatetimeIndex and base asset returns.

    Returns
    -------
    fmp_df: DataFrame, T x K, where T is the number of observations and K the number of factors
        DataFrame with DatetimeIndex and macro factor-mimicking portfolio returns.
    """

    # convert DataFrame to numpy for faster and easier matrix computation
    if isinstance(portfolio_weights_df, pd.DataFrame):
        W = portfolio_weights_df.to_numpy()
    # base asset returns
    if isinstance(base_assets_ret_df, pd.DataFrame):
        R = base_assets_ret_df.to_numpy()
    # compute FMPs
    FMP = W.T @ R.T
    FMP = FMP.T
    # create factor mimicking portfolio df
    fmp_df = pd.DataFrame(FMP, index=base_assets_ret_df.index, columns=[col.replace('yhat', 'fmp') for col in portfolio_weights_df.columns])

    return fmp_df

# compute macro factor mimicking portfolios from observable macro factors and base asset returns
def macro_fmp(obs_mf_df, base_assets_ret_monthly_df, base_assets_ret_daily_df, frequency='daily'):
    """
    Computes macro factor-mimicking portfolios from observable macro factors and base asset returns.
    Frequency can be set to monthly or daily, depending on the desired frequency of MFMP returns.

    Parameters
    ----------
    obs_mf_df: DataFrame
        DataFrame with DatetimeIndex and observale macro factors (surprises).
    base_assets_ret_monthly_df: DataFrame
        DataFrame with DatetimeIndex and monthly base asset returns.
    base_assets_ret_daily_df: DataFrame
        DataFrame with DatetimeIndex and daily base asset returns.

    Returns
    -------
    fmp_df: DataFrame, T x K, where T is the number of observations and K the number of factors
        DataFrame with DatetimeIndex and macro factor-mimicking portfolio returns at the selected frequency.
    """

    #orthogonalize macro factors
    omf_df = orthogonalize_factors(obs_mf_df, output_format='df')

    #set target vol
    mf_vol_adj_df = target_vol(omf_df)

    # estimate fitted macro factors
    fitted_mf_df = fitted_macro_factors(mf_vol_adj_df, base_assets_ret_monthly_df)

    # estimate factor betas
    factor_betas_df = factor_betas(base_assets_ret_monthly_df, fitted_mf_df)

    # compute portfolio weights
    portfolio_weights_df = portfolio_weights(factor_betas_df, output_format='df')

    # create macro factor-mimicking portfolios
    # if frequency daily
    if frequency == 'daily':
        fmp_df = create_fmp(portfolio_weights_df, base_assets_ret_daily_df)
    # if frequency monthly
    else:
        fmp_df = create_fmp(portfolio_weights_df, base_assets_ret_monthly_df)

    return fmp_df
