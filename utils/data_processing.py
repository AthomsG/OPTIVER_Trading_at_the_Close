import numpy as np
import pandas as pd

def feature_engineering(X):
    """
    Perform feature engineering on the input DataFrame.

    Parameters:
    - X: pandas DataFrame
        The input DataFrame containing the features.

    Returns:
    - pandas DataFrame
        The DataFrame with additional engineered features.
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    _X = X.copy()

    _X["liquidity_imbalance"] = _X.eval("(bid_size-ask_size)/(bid_size+ask_size)")
    _X["matched_imbalance"] = _X.eval("(imbalance_size-matched_size)/(matched_size+imbalance_size)")
    _X["price_spread"] = _X["ask_price"] - _X["bid_price"]
    _X['market_urgency'] = _X['price_spread'] * _X['liquidity_imbalance']

    # Replace 'NaN' with 0, in case of division by 0
    _X.fillna(0, inplace=True)

    return _X


def data_preprocessing(data, remove_missing=False):
    """
    Preprocess the input data by handling missing values and performing one-hot encoding.

    Parameters:
    - data: pandas DataFrame
        The input DataFrame containing the data.
    - remove_missing: bool, optional (default=False)
        If True, remove all rows with missing values.

    Returns:
    - DataFrame
        The preprocessed DataFrame.
    """
    if remove_missing:
        # Remove all rows with missing values
        data = data.dropna()

    else:
        # Replace missing values that correspond to undefined quantities with 0
        data['far_price']  = data['far_price'].fillna(0)
        data['near_price'] = data['near_price'].fillna(0)

        # Identify columns with missing values
        columns_with_missing = data.columns[data.isnull().any()]

        # Replace remaining missing values with variable median distribution
        for column in columns_with_missing:
            median_value = data[column].median()
            data[column] = data[column].fillna(median_value)

    # One-hot encode categorical variable 'imbalance_buy_sell_flag'
    data = pd.get_dummies(data, columns=['imbalance_buy_sell_flag'], prefix='imbalance_flag')

    # Rename the column 'imbalance_flag_-1' to 'imbalance_flag_neg_1' for readability
    data = data.rename(columns={'imbalance_flag_-1': 'imbalance_flag_neg_1'})
    
    return data
    

def remove_extreme_values(dataframe, columns, sigma=4):
    """
    Remove data points that are 4 standard deviations away from the mean in specified columns.

    Parameters:
    - dataframe: pandas DataFrame
        The input DataFrame containing the data.
    - columns: list of column names
        The list of column names in which extreme values will be removed.
    - sigma: int, optional (default=4)
        The number of standard deviations used to define the bounds for extreme values.

    Returns:
    - DataFrame
        A new DataFrame with extreme values removed from the specified columns.
    """

    # Copy the original DataFrame to avoid imputating original data
    df_copy = dataframe.copy()

    # Identify count variables by '_size' suffix
    count_variables = [column for column in columns if '_size' in column]

    # Apply square root transformation to count variables (helps with skewness)
    # for var in count_variables:
    #    df_copy[var] = np.sqrt(df_copy[var])
    
    # Iterate over each column and remove extreme values
    for column in columns:
        mean_value = df_copy[column].mean()
        std_value = df_copy[column].std()

        # Lower and upper bounds for extreme values
        lower_bound = mean_value - sigma * std_value
        upper_bound = mean_value + sigma * std_value

        # Remove rows with values outside the bounds
        df_copy = df_copy[(df_copy[column] >= lower_bound) & (df_copy[column] <= upper_bound)]

    return df_copy

def split_data(data, n):
    """
    Split the input data into training and test sets based on the specified date threshold.

    Parameters:
    - data: pandas DataFrame
        The input DataFrame containing the data.
    - n: int
        The date threshold for splitting the data.

    Returns:
    - tuple of DataFrames
        A tuple containing the training and test sets.
    """
    train = data[data['date_id'] < n]
    test = data[data['date_id'] >= n]
    return train, test