#imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from math import sqrt

from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from statsmodels.formula.api import ols



'''
*------------------*
|                  |
|     Evaluate     |
|                  |
*------------------*
'''


def split_zillow(df):
    """
    split_zillow will take one argument(df) and 
    then split our data into 20/80, 
    then split the 80% into 30/70
    
    perform a train, validate, test split
    
    return: the three split pandas dataframes-train/validate/test
    """  
    
    train_validate, test = train_test_split(df, test_size=0.2, random_state=3210)
    train, validate = train_test_split(train_validate, train_size=0.7, random_state=3210)
    return train, validate, test




def train_validate_test(df, target):
    '''
    this function takes in a dataframe and splits it into 3 samples, 
    a test, which is 20% of the entire dataframe, 
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe. 
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable. 
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test. 
    '''
    # split df into test (20%) and train_validate (80%)
    train, validate, test = split_zillow(df)
        
    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test

def get_numeric_X_cols(X_train, object_cols):
    '''
    takes in a dataframe and list of object column names
    and returns a list of all other columns names, the non-objects. 
    '''
    numeric_cols = [col for col in X_train.columns.values if col not in object_cols]
    
    return numeric_cols


def min_max_scale(X_train, X_validate, X_test, numeric_cols):
    '''
    this function takes in 3 dataframes with the same columns, 
    a list of numeric column names (because the scaler can only work with numeric columns),
    and fits a min-max scaler to the first dataframe and transforms all
    3 dataframes using that scaler. 
    it returns 3 dataframes with the same column names and scaled values. 
    '''
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).


    scaler = MinMaxScaler(copy=True).fit(X_train[numeric_cols])

    #scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train. 
    # 
    X_train_scaled_array = scaler.transform(X_train[numeric_cols])
    X_validate_scaled_array = scaler.transform(X_validate[numeric_cols])
    X_test_scaled_array = scaler.transform(X_test[numeric_cols])

    # convert arrays to dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled_array, 
                                  columns=numeric_cols).\
                                  set_index([X_train.index.values])

    X_validate_scaled = pd.DataFrame(X_validate_scaled_array, 
                                     columns=numeric_cols).\
                                     set_index([X_validate.index.values])

    X_test_scaled = pd.DataFrame(X_test_scaled_array, 
                                 columns=numeric_cols).\
                                 set_index([X_test.index.values])

    
    return X_train_scaled, X_validate_scaled, X_test_scaled



def create_dummies(df, object_cols):
    '''
    This function takes in a dataframe and list of object column names,
    and creates dummy variables of each of those columns. 
    It then appends the dummy variables to the original dataframe. 
    It returns the original df with the appended dummy variables. 
    '''
    
    # run pd.get_dummies() to create dummy vars for the object columns. 
    # we will drop the column representing the first unique value of each variable
    # we will opt to not create na columns for each variable with missing values 
    # (all missing values have been removed.)
    dummy_df = pd.get_dummies(df[object_cols], dummy_na=False, drop_first=True)
    
    # concatenate the dataframe with dummies to our original dataframe
    # via column (axis=1)
    df = pd.concat([df, dummy_df], axis=1)

    return df




def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # create a mask of columns whether they are object type or not
    mask = np.array((df.dtypes == "object") | (df.dtypes == "category"))

        
    # get a list of the column names that are objects (from the mask)
    object_cols = df.iloc[:, mask].columns.tolist()
    
    return object_cols



def prep4model(df, target):
    '''
    prep4model takes in a dataframe and target
    - produces a list of object column names
    - splits data into X/y train/validate/test
    - produces a list of numeric column names
    - scales the X_train/validate/test
    returns: split & scaled data.
    '''
    # get object columns names
    object_cols = get_object_cols(df)
    
    # split data 
    X_train, y_train, X_validate, y_validate, X_test, y_test = train_validate_test(df, target)
    
    # get numeric column names
    numeric_cols = get_numeric_X_cols(X_train, object_cols)

    # scale data 
    X_train_scaled, X_validate_scaled, X_test_scaled = min_max_scale(X_train, X_validate, X_test, numeric_cols)
    
    return X_train, X_train_scaled, y_train, X_validate, X_validate_scaled, y_validate, X_test, X_test_scaled, y_test




def select_kbest(X, y, n):
    '''
    select_kbest takes in the 
    predictors (X), 
    the target (y), and 
    the number of features to select (k) and 
    returns the names of the top k selected features based on the SelectKBest class
    '''
    
    # parameters: f_regression stats test
    f_selector = SelectKBest(f_regression, k= n)
    
    # find the top 2 X-feats correlated with y
    f_selector.fit(X, y)
    
    # boolean mask of whether the column was selected or not. 
    feature_mask = f_selector.get_support()
    
    # get list of top K features. 
    f_feature = X.iloc[:,feature_mask].columns.tolist()
    
    return f_feature





def rfe(X, y, n):
    '''
    rfe takes in the 
    predictors (X), 
    the target (y), and 
    the number of features to select (k) and 
    returns the names of the top k selected features based on the SelectKBest class
    '''
    
    # initialize the ML algorithm
    lm = LinearRegression()
    
    # create the rfe object, indicating the ML object (lm) and the number of features I want to end up with. 
    rfe = RFE(lm, n)
    
    # fit the data using RFE
    rfe.fit(X,y)  
    
    # get the mask of the columns selected
    feature_mask = rfe.support_
    
    # get list of the column names. 
    rfe_feature = X.iloc[:,feature_mask].columns.tolist()
    
    return rfe_feature





def plot_residuals(target, yhat):
    '''
    plot_residuals will take in a target series and prediction series
    and plot the residuals as a scatterplot.
    '''
    
    residual = target - yhat
    
    plt.scatter(target, residual)
    plt.axhline(y = 0, ls = ':')
    plt.xlabel("target")
    plt.ylabel("residual")
    plt.title('Residual Plot')
    plt.show
    
    
    
    
def regression_errors(target, yhat):
    '''
    regression_errors takes in a target and prediction series
    and prints out the regression error metrics.
    '''
    residual = target - yhat
    
    mse = mean_squared_error(target, yhat)
    sse = (residual **2).sum()
    rmse = sqrt(mse)
    tss = ((target - yhat.mean()) ** 2).sum()
    ess = ((yhat - target.mean()) ** 2).sum()
    print(f"""
    MSE: {round(mse,2)}
    SSE: {round(sse,2)}
    RMSE: {round(rmse,2)}
    TSS: {round(tss,2)}
    ESS: {round(ess,2)}
    """)
    
    
    
    
    
def baseline_mean_errors(target):
    '''
    baseline_mean_errors takes in a target 
    and prints out the regression error metrics for the baseline.
    '''
    baseline = target.mean()
    
    residual = target - (baseline)
    
    sse_baseline = (residual **2).sum()
    mse_baseline = sse_baseline / len(target)
    rmse_baseline = sqrt(mse_baseline)
    
    print(f"""
    MSE_baseline: {round(mse_baseline,2)}
    SSE_baseline: {round(sse_baseline,2)}
    RMSE_baseline: {round(rmse_baseline,2)}
    """)
    
    
    
    
    
def baseline_median_errors(target):
    '''
    baseline_mean_errors takes in a target 
    and prints out the regression error metrics for the baseline.
    '''
    baseline = target.median()
    
    residual = target - (baseline)
    
    sse_baseline = (residual **2).sum()
    mse_baseline = sse_baseline / len(target)
    rmse_baseline = sqrt(mse_baseline)
    
    print(f"""
    MSE_baseline: {round(mse_baseline,2)}
    SSE_baseline: {round(sse_baseline,2)}
    RMSE_baseline: {round(rmse_baseline,2)}
    """)    
    
    
    
def better_than_baseline(target, yhat):
    '''
    better_than_baseline takes in a target and prediction 
    and returns boolean answering if the model is better than the baseline.
    '''
    
    rmse_baseline = sqrt((((target - (target.mean())) **2).sum()) * len(target))
    rmse_model = sqrt((((target - yhat) **2).sum()) * len(target))
    return rmse_model < rmse_baseline




def model_significance(ols_model):
    return {
        'r^2 -- variance explained': ols_model.rsquared,
        'p-value -- P(data|model == baseline)': ols_model.f_pvalue,
    }




def residuals(actual, predicted):
    return actual - predicted

def sse(actual, predicted):
    return (residuals(actual, predicted) **2).sum()

def mse(actual, predicted):
    n = actual.shape[0]
    return sse(actual, predicted) / n

def rmse(actual, predicted):
    return sqrt(mse(actual, predicted))

def ess(actual, predicted):
    return ((predicted - actual.mean()) ** 2).sum()

def tss(actual):
    return ((actual - actual.mean()) ** 2).sum()





def reg_error_metrics(target, yhat):
    '''
    reg_error_metrics takes in target and prediction series 
    and returns a dataframe that contains the SSE/MSE/RMSE metrics 
    for. both model and baseline
    and answers if the model is better than the baseline.
    '''
    
    df = pd.DataFrame(np.array(['SSE', 'MSE','RMSE']), columns=['metric'])
    
    df['model_error'] = np.array([sse(target, yhat),  mse(target, yhat), rmse(target, yhat)])
    
    df['baseline_error'] = np.array([sse(target, target.mean()), mse(target, target.mean()), rmse(target, target.mean())])
    
    df['better_than_baseline'] = df.baseline_error > df.model_error
    
    df = df.set_index("metric")
    
    return df