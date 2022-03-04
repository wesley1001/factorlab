# import libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from ppca import PPCA
from sklearn.linear_model import LassoLarsIC, Lasso
from sklearn.metrics import r2_score
import statsmodels.api as sm
from scipy.stats import ttest_1samp, chi2_contingency, spearmanr, kendalltau, contingency

# principal component analysis
def pca(features_df, method='ppca', pc_name=None, n_components=1, min_var_explained=0.9):

    """
    Principal component analysis. Dimensionality reduction technique which converts a set of correlated features to a smaller
    number of uncorrelated features (principal components). Makes use of scikit-learn package for PCA and PPCA for
    probabilistic PCA.

    Parameters
    ----------
    features_df: DataFrame
        DataFrame with DatetimeIndex and features.
    pc_name: str, default None
        Name/prefix for principal components.
    n_components: int, default 1 for pca, 2 for ppca
        Number of principal components to keep.
    min_var_explained: float
        Minimum variance to be explained by principal components. Additional principal components will be automatically added
        until this minimum threshold is reached.

    Returns
    -------
    features_df: DataFrame
        DataFrame with DatetimeIndex, features and principal components added to it.
    """

    # probabilistic pca
    if method == 'ppca':

        # min nunber of components for PPCA
        n_components=2

        # drop NaNs
        features_df.dropna(how='all', inplace=True)

        # fit pca
        ppca = PPCA()
        ppca.fit(features_df.to_numpy(), d=n_components)
        pcs = ppca.transform()

        # variance explained by first n components
        variance_explained = ppca.var_exp[-1]

        # keep adding PCs until min var explained threhsold is met
        while variance_explained < min_var_explained:

            # add component to pca
            n_components += 1

            # fit pca
            ppca = PPCA()
            ppca.fit(features_df.to_numpy(), d=n_components)
            pcs = ppca.transform()

            # variance explained by first n components
            variance_explained = ppca.var_exp[-1]

        # print number of PCs added
        print('{} principal components added'.format(len(ppca.var_exp)))

    # pca
    else:

        # drop NaNs
        features_df.dropna(inplace=True)

        # fit pca
        pca = PCA(n_components=n_components)
        pcs = pca.fit_transform(features_df)

        # variance explained by first n components
        variance_explained = pca.explained_variance_ratio_.cumsum()[-1]

        # keep adding PCs until min var explained threhsold is met
        while variance_explained < min_var_explained:

            # add component to pca
            n_components += 1

            # fit pca
            pca = PCA(n_components=n_components)
            pcs = pca.fit_transform(features_df)

            # variance explained by first n components
            variance_explained = pca.explained_variance_ratio_.cumsum()[-1]

        # print number of pcs added
        print('{} principal components added'.format(len(pca.explained_variance_ratio_)))

    # add pcs to features df
    for i in range(pcs.shape[1]):
        if pc_name is not None:
            features_df[pc_name + '_pc'+ str(i+1)] = pcs[:, i]
        else:
            features_df['pc'+ str(i+1)] = pcs[:, i]

    # print variance explained by PCs
    print("Variance explained: {}\n".format(round(variance_explained,4)))

    return features_df

# orthogonalization of correlated factors in a multifactor model
def orthogonalize_factors(factors_df, output_format='df'):

    """
    As described by Klein and Chow (2013) in Orthogonalized Factors and Systematic Risk Decompositions:
    https://www.sciencedirect.com/science/article/abs/pii/S1062976913000185
    They propose an optimal simultaneous orthogonal transformation of factors, following the so-called symmetric procedure
    of Schweinler and Wigner (1970) and Löwdin (1970).  The data transformation allows the identification of the underlying uncorrelated
    components of common factors without changing their correlation with the original factors. It also facilitates the systematic risk
    decomposition by disentangling the coefficient of determination (R²) based on factors' volatilities, which makes it easier to distinguish
    the marginal risk contribution of each common risk factor to asset returns.

    Parameters
    ----------
    factors_df: DataFrame
        DataFrame with DatetimeIndex and facotrs to orthogonalize
    output_format: str,{'array', 'df'}, defaul 'df'
        Select the output format, numpy.array or DataFrame.

    Returns
    -------
    F_orth: numpy.array or DataFrame
        numpy.array of orthogonalized factors or DataFrame with DatetimeIndex and orthogonalized factors.
    """

    # before orthogonalization, factors should be normalized
    # if factors are in a DataFrame, convert to numpy for speed and ease of computation
    if isinstance(factors_df, pd.DataFrame):
        F = factors_df.to_numpy()
    # compute cov matrix
    M = np.cov(F.T, bias=False)
    # factorize cov matrix M
    u, s, vh = np.linalg.svd(M)
    # solve for symmetric matrix
    S = u @ np.diag(s**(-0.5)) @ vh
    # rescale symmetric matrix to original variances
    S_rs = S  @ (np.diag(np.sqrt(M)) * np.eye(S.shape[0], S.shape[1]))
    # convert to orthogonalized matrix
    F_orth = F @ S_rs
    # if selected, convert output format back to dataframe
    if output_format == 'df':
        F_orth = pd.DataFrame(F_orth, index=factors_df.index, columns=factors_df.columns)

    return F_orth

# multifactor risk model regressions and output tables (df)
def factor_exposures(ret_df, factors_df):

    """
    Multifactor risk model regressions and output.
    Multivariate regressions which iterates through individual asset returns (columns in ret_df) on factor returns
    (factors_df). Output for each asset is provided in a DataFrame (table_df) and stored in a dictionary of DataFrames (tables_dict)
    for all asset returns in ret_df. Ideally, factors should be orthogonalized before running this regression to improve
    statistical properties and interpretability. See orthogonalize_factors.py for the orthongonalization algorithm.

    Parameters
    ----------
    ret_df: DataFrame
        DataFrame with DatetimeIndex and asset returns, each column is an independent variable in a multivariate regression on the factor returns.
    factors_df: DataFrame
        DataFrame with DatetimeIndex and factor returns, these are the explanatory variables in the multifactor risk model regressions.

    Returns
    -------
    tables_dict: Dictionary of DataFrames
        Dictionary of DataFrames. Regression output for each asset is stored under the asset ticker in the dictionary;
        e.g. tables_dict['BTC'] provides the output for the regression of Bitcoin on the factors in factors_df.
        Output includes beta estimates (coefficients), t-statistics, p-values, r-squared (for all factors) and decomposed r-squared (for each factor).
    """

    # create df dict
    tables_dict = {}

    # iterate through asset cols
    for col in ret_df.columns:

        # stats to be included in table output
        stats = ['beta', 't_stat', 'p_val', 'var_expl']
        # regress asset returns on predictors
        data = pd.concat([ret_df.loc[:,[col]], factors_df], join='outer', axis=1)
        y, X = data.iloc[:,0], data.iloc[:,1:]
        # add constant
        X = sm.add_constant(X)
        # specify model
        model = sm.OLS(y,X, missing='drop')
        # fit model
        res = model.fit()
        # get stats
        rsq, betas, t_stats, p_vals = res.rsquared, res.params, res.tvalues, res.pvalues
        # get std
        f_sigma, a_sigma = X.std(), y.std()
        # compute risk contribution & r-square
        r_sq = ((betas * f_sigma)**2) / a_sigma**2
        # add to dictionary
        stats = {'beta':betas, 't_stat':t_stats, 'p_val': p_vals, 'var_expl': r_sq}
        # create df table with dictionary data
        table_df = pd.DataFrame(stats).T
        # add rsq as last col
        table_df[''] = [np.nan, np.nan, np.nan, rsq]
        # rename const col
        table_df.rename(columns={'const':'exp_ret'}, inplace=True)
        # add to tables dict and round
        tables_dict[col] = table_df.round(decimals=4)

    return tables_dict

# Lasso regression of a target on features to be used for feature selection
def lasso_feature_selection(target, features_df, alpha=0.05, auto_selection=True, criterion='aic'):
    """
    LASSO supervised learning feature selection, used as a preliminary step in the targeted PCA algorithm.
    Selects a subset of relevant features from a broader set of features by removing the redundant
    or irrelevant features, or features which are strongly correlated in the data without much loss of information.

    Parameters
    ----------
    target: Series or DataFrame
        Series or DataFrame with DatetimeIndex and target variable (y).
    features_df: DataFrame
        DataFrame with datetimeIndex and features (X).
    alpha: float, default 0.05
        Constant that multiplies the L1 regularization term. Defaults to 1.0. Alpha = 0 is equivalent to an OLS regression.
    auto-selection: bool, default True
        Lasso model fit with Lars using BIC or AIC for model selection.
    criterion: str, {'aic', 'bic'}, default 'aic'
        AIC is the Akaike information criterion and BIC is the Bayes Information criterion. Such criteria are useful to
        select the value of the regularization parameter by making a trade-off between the goodness of fit and
        the complexity of the model.A good model should explain the data well while being simple.

    Returns
    -------
    selected_features: list
        List of the subset of selected features from the LASSO regression.
    """

    # if target is Series, convert to DataFrame
    if isinstance(target, pd.Series):
        target = target.to_frame()

    # create reg df for lasso regression and feature selection
    reg_df = target.merge(features_df, how='outer', left_index=True, right_index=True).dropna()

    # create target and predictors
    # y first col, X rest of df
    X, y = reg_df.iloc[:, 1:], reg_df.iloc[:, 0]

    # auto selection
    if auto_selection==True:
        # specify model
        model = LassoLarsIC(criterion=criterion, normalize=False)
    else:
        model = Lasso(alpha=alpha)

    # fit model
    model.fit(X,y)
    print('Regressing {} on asset returns\n'.format(y.name))
    # predictions y_hat
    y_hat = model.predict(X)
    # compute adj R^2
    r2 = r2_score(y, y_hat)
    print('Adjusted R^2: {} \n'.format(round(r2, 2)))
    # selected features
    estimated_coef = np.nonzero(model.coef_)
    coef_idxs = estimated_coef[0].tolist()
    selected_features = X.iloc[:, coef_idxs].columns.tolist()
    print('{} features were selected: {}\n'.format(len(selected_features), selected_features))
    # compute removed features percentage
    percent_removed_features = 100 - (round(len(coef_idxs) / X.shape[1], 2) * 100)
    print('{}% of features were removed\n'.format(percent_removed_features))

    return selected_features

# targeted PCA algorithm
def tpca(target, features_df, min_var_explained=0.9):
    """
    Targeted PCA which uses LASSO supervised learning feature selection as a preliminary step before PCA.
    Selects a subset of relevant features from a broader set of features by removing the redundant or
    irrelevant features, i.e. features which are strongly correlated in the data without much loss of information.
    Runs a PCA on the subset of selected features from the LASSO regression.
    See Forecasting economic time series using targeted predictors by Bai and Ng (2008) for details:
    https://www.sciencedirect.com/science/article/abs/pii/S0304407608001085

    Parameters
    ----------
    target: Series or DataFrame
        Series or DataFrame with DatetimeIndex and target variable (y).
    features_df: DataFrame
        DataFrame with DatetimeIndex and features (X).
    min_var_explained: float, default 0.9
        Minimum variance explained by first L principal components.

    Returns
    -------
    pcs_df:
        DataFrame with DatetimeIndex and the first L principal components.
    """

    # if target is Series, convert to DataFrame
    if isinstance(target, pd.Series):
        target = target.to_frame()

    # create df for lasso regression and feature selection
    df = target.merge(features_df, how='outer', left_index=True, right_index=True).dropna()

    # add lags
    lasso_df = add_lags(df)

    # select features from LASSO regression
    selected_features = lasso_feature_selection(lasso_df.iloc[:, 0], lasso_df.iloc[:,1:])

    # extract L pcs from with selected features from lasso
    pcs_df = pca(lasso_df.loc[:, selected_features], pc_name='tpca', method='ppca', min_var_explained=min_var_explained).loc[:, 'tpca_pc1':]

    return pcs_df

# project target on predictors to estimate fitted target values (y-hat)
def project_target(target, features_df):
    """
    Project target variable on the features to obtain the fitted values for the target variable. Can be used for
    instrumental variables estimation.

    Parameters
    ----------
    target: Series
        Series with DatetimeIndex and target variable (y).
    features_df: DataFrame
        DataFrame with DatetimeIndex and features (X).

    Returns
    -------
    fitted_target: Series
        Series with DatetimeIndex and fitted values for the target variable.
    """

    # convert Series to DataFrame
    if isinstance(target, pd.Series):
        target = target.to_frame()

    # create df for lasso regression and feature selection and drop NaNs
    df = target.merge(features_df, how='outer', left_index=True, right_index=True).dropna()

    # X: predictors, y: target
    X, y = df.iloc[:,1:], df.iloc[:,0]
    # add constant
    X = sm.add_constant(X)

    # specify model
    model = sm.OLS(y,X)
    # fit model
    res = model.fit()
    # y-hat
    fitted_target = res.predict(X)
    # print regression details
    print('Project {} onto the features'.format(y.name))
    # print output
    print(res.summary())

    return fitted_target

# Information coefficient (IC)
def IC(factors, target_ret, lookahead=14, pc1=True, factor_bins=5, target_bins=5, ic_rolling_window=365):
    """
    Calculates correlation for returns, or degree of association for labels (bins), between the alpha factors (features)
    and forward returns (target). Correlation measures in what way two variables are related, whereas, association measures
    how related the variables are.
    orrelation measures such as the spearman rank, kendall and pearson compute.

    Measures the degree to which two nominal or ordinal variables are related, or the level of their association.
    Both factors and target should be discretized before computing a measure of the degree to which category membership

    Parameters
    ----------
    factors: Series or DataFrame
        Series or DataFrame with DatetimeIndex and alpha factors.
    target: Series
        Series with DatetimeIndex and target variable.
    lookahead: int, default 1
        Number of periods to shift forward returns (target).
    factor_bins: int, optional, default None
        Number of bins into which to discretize/label the normalized factors. None leaves factor inputs unchanged.
    target_bins: int, optional, default None
        Number of bins into which to discretize/label the normalized target. None leaves target inputs unchanged.

    Returns
    -------
    metrics: DataFrame
        DataFrame with computed stasticial association/correlation metrics.
    """

    # if bins is None or 1
    if factor_bins < 2 or target_bins < 2:
        print("Number of bins must be larger than 1. Please increase number of bins.\n")
        return

    else:

        # if factors or target are Series, convert to DataFrame
        if isinstance(factors, pd.Series):
            factors = factors.to_frame()
        if isinstance(target_ret, pd.Series):
            target_ret = target_ret.to_frame()
        # get principal components of factors
        if pc1:
            factors = pca(factors, method='pca', pc_name='trend', min_var_explained=0.5)
            # constrain pc1 to be positively correlated with factors
            col = factors.loc[:, factors.columns.str.contains('pc1')].columns[0]
            if factors.corr()[col].mean() < 0:
                factors[col] = factors[col] * -1

        # discretize factors and target
        # factor bins
        factor_quantiles_df = discretize(factors, bins=factor_bins)
        # target bins
        target_quantiles_df = discretize(target_ret, bins=target_bins)

        # merge factors and target, shift target by lookahead periods
        df = factors.merge(target_ret.shift(lookahead * -1), how='outer', left_index=True, right_index=True).dropna()
        quantiles_df = factor_quantiles_df.merge(target_quantiles_df.shift(lookahead * -1), how='outer',
                                                 left_index=True, right_index=True).dropna()

        # create empy dfs for correlation measures
        metrics, ic_df = pd.DataFrame(index=factors.columns), pd.DataFrame(index=df.index, columns=factors.columns)

        # calculate correlation and assocation measures
        # loop through factors
        for col in factors.columns:
            # contingency table
            cont_table = pd.crosstab(quantiles_df[col], quantiles_df.iloc[:, -1])
            # add metrics
            metrics.loc[col, 'IC/spearman_rank'] = spearmanr(quantiles_df[col], df.iloc[:, -1])[0]
            metrics.loc[col, 'p-val'] = spearmanr(quantiles_df[col], df.iloc[:, -1])[1]
            metrics.loc[col, 'kendall_tau'] = kendalltau(quantiles_df[col], df.iloc[:, -1])[0]
            metrics.loc[col, 'cramer_v'] = contingency.association(cont_table, method='cramer')
            metrics.loc[col, 'tschuprow_t'] = contingency.association(cont_table, method='tschuprow')
            metrics.loc[col, 'pearson_cc'] = contingency.association(cont_table, method='pearson')
            metrics.loc[col, 'chi2'] = chi2_contingency(cont_table)[0]
            metrics.loc[col, 'autocorrelation'] = \
            spearmanr(quantiles_df[col].iloc[1:].dropna(), quantiles_df[col].shift(1).dropna())[0]

            # window size
            window_size = ic_rolling_window

            # while loop for rolling window spearman rank corr
            while window_size <= df.shape[0]:
                # compute spearman rank correlation
                ic_df[col].iloc[window_size - 1] = \
                spearmanr(quantiles_df[col].iloc[window_size - ic_rolling_window:window_size],
                          df.iloc[window_size - ic_rolling_window:window_size, -1])[0]
                window_size += 1

    # plot ic df
    plt.style.use('ggplot')
    ic_df.plot(legend=True, figsize=(15, 7), linewidth=2, rot=0, title='Information Coefficient',
               ylabel='{}-day rolling window'.format(ic_rolling_window));

    # create dict to store dfs
    dict_dfs = {'metrics': metrics.sort_values(by='IC/spearman_rank', ascending=False).round(decimals=4),
                'ic_rolling': ic_df.dropna()}

    return dict_dfs

# factor returns
def factor_returns(factors, returns, lookahead=14, pc1=True, bins=None, tails=None, tcost=None):
    """
    Screens features for predictive relationship with the target and provides summary performance statistics

    Parameters
    ----------
    factors: Series or Dataframe
        Series or DataFrame with DatetimeIndex and factors.
    returns: Series
        Target returns series.
    lookahead: int, default 1
        Number of periods to shift forward returns.
    pc1: bool, default False
        Compute principal components of factors and add them to factors.
    bins: int, default None
        Number of desired bins for discretization.
    tails: str, default None
        Keeps only tail bins and ignores middle bins, 'two' for both tails, 'left' for left, 'right' for right
    tcost: float, default None
        Transaction fee subtracted from returns to get net returns.
        Depends on exchange, e.g. Binance maker/taker fee is 0.001.

    Returns
    -------
    dict_dfs: dictionary with DataFrames
        'net_ret' DataFrame with returns (net of t-cost) of target returns scaled on factor signals;
        'perf' DataFrame with performance metrics of net returns;
    """

    # convert to df if series
    if isinstance(factors, pd.Series):
        factors = factors.to_frame()

    # get principal components of factors
    if pc1:
        factors = pca(factors, method='pca', pc_name='trend', min_var_explained=0.5)
        # constrain pc1 to be positively correlated with factors
        col = factors.loc[:, factors.columns.str.contains('pc1')].columns[0]
        if factors.corr()[col].mean() < 0:
            factors[col] = factors[col] * -1

    # convert factors to signal
    signal_df = (normalize(factors, method='percentile') * 2) - 1
    # discretize signal into signal quantiles between -1 and 1
    signal_quantiles_df = discretize(signal_df, bins=bins, signal=True, tails=tails)

    # compute factor returns and tcosts
    if tcost is None:
        tcost = 0
    if bins is None:
        ret_df = signal_df.shift(lookahead).multiply(returns, axis=0)
        tcost_df = abs(signal_df.diff()).shift(lookahead) * tcost * (1 / lookahead)
    else:
        ret_df = signal_quantiles_df.shift(lookahead).multiply(returns, axis=0)
        tcost_df = abs(signal_quantiles_df.diff()).shift(lookahead) * tcost * (1 / lookahead)

    # compute net ret
    if lookahead > 1:
        net_ret_df = (ret_df / lookahead) - tcost_df
    else:
        # compute net returns
        net_ret_df = ret_df - tcost_df
    # create performance metrics df for net returns
    perf_df = factor_performance(net_ret_df, returns)
    perf_df.index.name = 'alpha_factors'

    # create quantiles for mean return by quantile plot
    if bins is None:
        bins = 5
    factor_quantiles_df = (discretize(factors, bins=bins, tails=tails) + 1).astype(int)

    # compute IR for each bin
    bins_ret = pd.DataFrame(index=range(1, bins + 1))
    for col in net_ret_df.columns:
        bins_ret[col] = (net_ret_df[col].groupby(factor_quantiles_df[col].shift(lookahead)).mean())
    # name index quantile
    bins_ret.index.name = 'quantile'
    # add top vs bottom quantile bin in index
    bins_ret.loc['top vs. bottom', :] = bins_ret.iloc[-1] - bins_ret.iloc[0]

    # show cum ret and bar chart subplots
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    net_ret_df.cumsum().plot(legend=True, linewidth=2, rot=0, ax=ax1, title='Cumulative returns')
    ax1.set_ylabel('Cumulative returns (net)');

    # plot the mean returns by quantile of the best performing factor
    col = perf_df.index[0]
    bins_ret[col].plot(kind='bar', color='#C59B8E', legend=False, rot=90, ax=ax2,
                       title='Mean Returns (net) by Factor Quantile: {}'.format(col));
    ax2.set_ylabel('Mean returns (net)');

    # create dict to store dfs
    dict_dfs = {'net_ret': net_ret_df, 'perf': perf_df}

    return dict_dfs

# factor performance
def factor_performance(factor_ret, returns, freq='daily'):
    """
    Computes key performance metrics for factor returns.

    Parameters
    ----------
    returns: Series or DataFrame
        Series or DataFrame with DatetimeIndex and factor returns series.
    freq: str, {'min', 'hourly', 'daily_business', 'daily', 'weekly', 'monthly'}, default 'daily'
        Frequency of returns.

    Returns
    -------
    metrics: DataFrame
        DataFrame with computed performance metrics.
    """

    # annualizaton adjustment factor
    if freq == 'min':
        ann_adj = 365 * 24 * 60
    elif freq == 'hourly':
        ann_adj = 365 * 24
    elif freq == 'daily_business':
        ann_adj = 252
    elif freq == 'weekly':
        ann_adj = 52
    elif freq == 'monthly':
        ann_adj = 12
    else:
        ann_adj = 365

    # convert to df if series
    if isinstance(factor_ret, pd.Series):
        returns = factor_ret.to_frame()

    # create metrics df and add performance metrics
    metrics = pd.DataFrame(index=factor_ret.columns)
    metrics['Annual return'] = factor_ret.mean() * ann_adj
    metrics['Annual volatility'] = factor_ret.std() * np.sqrt(ann_adj)
    metrics['Sharpe ratio'] = (factor_ret.mean() / factor_ret.std()) * np.sqrt(ann_adj)
    metrics['Sortino ratio'] = (factor_ret.mean() / factor_ret[factor_ret < 0].std()) * np.sqrt(ann_adj)
    metrics['Skewness'] = factor_ret.skew()
    metrics['Kurtosis'] = factor_ret.kurt()
    metrics['P-val'] = ttest_1samp(factor_ret.dropna(), popmean=0)[1] / 2

    # loop through df
    for col in factor_ret.columns:

        y = factor_ret[col]
        X = sm.add_constant(returns, prepend=True)
        data = pd.concat([y, X], axis=1).dropna()
        # Fit and summarize OLS model
        res = sm.OLS(data.iloc[:,0], data.iloc[:,1:]).fit(missing='drop')
        # add to metrics
        metrics.loc[col,'Annual alpha'], metrics.loc[col, 'Beta'] = ((res.params[0]+1)**ann_adj-1), res.params[1]

    # sort by sharpe ratio and round values to 2 decimals
    metrics = metrics.sort_values(by='Sharpe ratio', ascending=False).astype(float).round(decimals=2)

    return metrics