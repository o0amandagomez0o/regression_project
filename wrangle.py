import pandas as pd
import numpy as np
import os
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from env import host, user, password


'''
*------------------*
|                  |
|     ACQUIRE      |
|                  |
*------------------*
'''

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

    
    


def zillow_df():
    '''
    This function reads in the zillow data from the Codeup db
    and returns a pandas DataFrame with only requested columns: 
    - square feet of the home 
    - number of bedrooms
    - number of bathrooms
    requested timeframe and sgl unit households.
    '''
    
    sql_query = """
    select parcelid, calculatedfinishedsquarefeet, bedroomcnt, bathroomcnt, taxvaluedollarcnt
    from properties_2017
    join predictions_2017 using(parcelid)
    where propertylandusetypeid IN ('260', '261', '263', '264', '265', '266', '269', '275')
    and unitcnt = 1
    and transactiondate between "2017-05-01" and "2017-08-31"
    """
    return pd.read_sql(sql_query, get_connection('zillow'))





def zillow_full():
    '''
    This function reads in the zillow data from the Codeup db
    and returns a pandas DataFrame with only requested columns: 
    - square feet of the home 
    - number of bedrooms
    - number of bathrooms
    requested timeframe and sgl unit households.
    '''
    
    query = """
    select *
    from properties_2017
    join predictions_2017 using(parcelid)
    """
    return pd.read_sql(query, get_connection('zillow'))

    
    


def get_zillow():
    '''
    This function reads in the zillow data from the Codeup db
    and returns a pandas DataFrame with ALL columns, 
    requested timeframe and sgl unit households.
    '''
    
    query = """
    select *
    from properties_2017
    join predictions_2017 using(parcelid)
    where propertylandusetypeid IN ('260', '261', '263', '264', '265', '266', '269', '275')
    and unitcnt = 1
    and transactiondate between "2017-05-01" and "2017-08-31";
    """
    return pd.read_sql(query, get_connection('zillow'))




'''
*------------------*
|                  |
|     PREPARE      |
|                  |
*------------------*
'''
def clean_zillow(df):
    '''
    clean _zillow will take in a dataframe, clean the data by renaming columns, establishing parcelid as index, drops all null/NaN rows, as well as removes home_value outliers.
    '''
    
    df = df.rename(columns={"bedroomcnt": "bedrooms", "bathroomcnt": "bathrooms", "calculatedfinishedsquarefeet": "square_feet", "taxamount": "taxes", "taxvaluedollarcnt": "home_value"})
    df = df.set_index("parcelid")
    df = df.dropna()
    
    upper_bound, lower_bound = outlier(df, "home_value", 1.5)
    
    df = df[df.home_value < upper_bound]
    
    return df





def clean_z2(df):
    '''
    clean _zillow will take in a dataframe, clean the data by:
    - changing the dtype of some features
    - renaming columns
    - establishing parcelid as index
    - drops all null/NaN rows
    - removes home_value outliers
    - creates dummy col for homes that have 3 bedrooms
    - drops unnecessary columns
    returns cleaned df
    '''
    
    df = df.rename(columns={"bedroomcnt": "bedrooms", "bathroomcnt": "bathrooms", "calculatedfinishedsquarefeet": "square_feet", "taxamount": "taxes", "taxvaluedollarcnt": "home_value"})

    df = df.set_index("parcelid")
    df = df.dropna()
    
    df = df.astype({'fips': 'int64', 'yearbuilt': 'int64'})
    
    upper_bound, lower_bound = outlier(df, "home_value", 1.5)
    
    df = df[df.home_value < upper_bound]
    
    df['bdrm_3'] = (df['bedrooms']).apply(lambda x: 1 if x == 3 else 0)
    
    dropstrcol = ['county', 'state']
    df = df.drop(columns=dropstrcol)
    
    
    return df
    
    


    
def outlier(df, feature, m):
    '''
    outlier will take in a dataframe's feature:
    - calculate it's 1st & 3rd quartiles,
    - use their difference to calculate the IQR
    - then apply to calculate upper and lower bounds
    - using the `m` multiplier
    '''
    q1 = df[feature].quantile(.25)
    q3 = df[feature].quantile(.75)
    
    iqr = q3 - q1
    
    multiplier = m
    upper_bound = q3 + (multiplier * iqr)
    lower_bound = q1 - (multiplier * iqr)
    
    return upper_bound, lower_bound
 
    
    
    
    
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





def wrangle_zillow():
    '''
    wrangle_zillow will: 
    - read in zillow dataset for transaction dates between 05/2017-08/2017 as a pandas DataFrame,
    - clean the data
    - split the data
    return: the three split pandas dataframes-train/validate/test
    '''
    
    df = clean_zillow(zillow_df())
    return split_zillow(df)




def scale_zillow(train, validate, test):
    '''
    scale_zillow will 
    - fits a min-max scaler to the train split
    - transforms all three spits using that scaler. 
    returns: 3 dataframes with the same column names and scaled values. 
    '''
    
    scaler = sklearn.preprocessing.MinMaxScaler()
    
    # Note that we only call .fit with the TRAINING data,
    scaler.fit(train)
    
    # but we use .transform to apply the scaling to all the data splits.    
    train_scaled = scaler.transform(train)
    validate_scaled = scaler.transform(validate)
    test_scaled = scaler.transform(test)
    
    # convert to arrays to pandas DFs
    train_scaled = pd.DataFrame(train_scaled, columns=train.columns)
    validate_scaled = pd.DataFrame(validate_scaled, columns=train.columns)
    test_scaled = pd.DataFrame(test_scaled, columns=train.columns)
    
    return train_scaled, validate_scaled, test_scaled