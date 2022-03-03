# import libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from ppca import PPCA
from sklearn.linear_model import LassoLarsIC, Lasso
from sklearn.metrics import r2_score
import statsmodels.api as sm

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

# evaluation metrics

