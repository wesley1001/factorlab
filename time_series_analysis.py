# import libraries
import pandas as pd
import numpy as np

# create lag variables
def add_lags(features_df, n_lags=24):
    """
    Add lags to features before running a distributed lag model regression.

    Parameters
    ----------
    features_df: DataFrame
        DataFrame with DatetimeIndex and features, the lags of which are to be created.
    n_lags: int, default 24
        Number of lags to be created.

    Returns
    -------
    features_df: DataFrame
        DataFrame with the orginal features and the lags of those features added.
    """

    # col list
    cols = features_df.columns.to_list()
    # iterate through each col
    for col in cols:
    # iterate through range of lags
        for i in range(n_lags):
            # create new col with lags
            features_df[col + '_l' + str(i+1)] = features_df[col].shift(i+1)

    # print statement with number of lags added
    print('{} lags were added to the features DataFrame for the distributed lag model regression\n'.format(n_lags))

    return features_df
