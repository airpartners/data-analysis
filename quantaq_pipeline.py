import quantaq
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import pickle

"""
https://github.com/scott-hersey/EB_AQ_Network/blob/master/initial_analysis_walkthrough.Rmd
https://quant-aq.github.io/py-quantaq/usage.html
"""

# constants
TOKEN_PATH = "token.txt"
sensor = "SN000-049" #this is a winthrop sensor, not a revere sensor for now
TODAY = datetime.today()
final_cols = ["timestamp", "timestamp_local", "temp_box", "temp_manifold", "rh_manifold", "pressure", "noise", "solar", "wind_dir", "wind_speed", "co", "no", "no2", "o3", "pm1", "pm25", "pm10", "co2"]
raw_vars = ["timestamp", "bin0", "bin1", "bin2", "bin3", "bin4", "bin5", "no_ae", "co_ae", "no2_ae"]
CUTOFF = 300

#running list of questions:
# do we need to do the same baseline correction function for the revere dataset, how will we know?
# why are values over 300 set to 0? are they considered sensor errors?
# when we check the flags, is that the only data we care about (the spikes)? or is it primarily a sanity check?
# what is igor and what does it provide that is different than the raw data?
#   - after getting data back from igor we check the time stamps between spikes again, how is that different
#     than the checked flags that we looked at before sending to igor?
# what is purpose/difference between raw data and the other data? do we need both in the same dataset or just one or another?
    # if we need both then we can request one after another and then join them as a df, then order by ASC timestamp
#convert wind speed from mph to m/s? most likely not needed
#list of dataframes? what was the separation, just double check that everything works so far
#auxiliary electrode? not having permission to view raw data for some reason
#there was something about filling in data when the time intervals were irregular or too infrequent

def request_data(serial_num, start_date=TODAY-timedelta(days=2), end_date=TODAY, raw=False):
    """
    Request data from QuantAQ's API.
    
    :param serial_num: (str) serial number of the sensor
    :param start_date: (datetime) datetime object representing beginning of date range to download data for
    :param end_date: (datetime) represents end of date range to download data for. Note that end_date is EXCLUSIVE of the last day,
    so end date of 2020-01-03 will return data up until 2020-01-02 at 11:59pm.
    :param raw: (bool) True if requesting raw data, False otherwise
    :returns: pandas Dataframe containing data
    """
    #convert datetime start, end to strings
    start = start_date.strftime("%Y-%m-%d")
    stop = end_date.strftime("%Y-%m-%d")

    s = datetime.now().second
    #perform QuantAQ request
    data = client.data.list(sn=serial_num, start=start, stop=stop, raw=raw)
    print(f"fetching data took {datetime.now().second-s} secs")

    #convert returned info to pandas df
    return pd.DataFrame(data)

def cleaning_final(df):
    #keep only the relevant columns:
    df = df[final_cols]
    #convert all timestamp strings to datetime objects
    #timestamp is already in UTC
    df = df.assign(timestamp=pd.to_datetime(df['timestamp']))
    # timestamp_local contains local time (but expresses it in UTC, so 18:59 Eastern is expressed as 18:59 UTC). need to change
    # the timezone without altering the hour of day.
    # So, convert to datetime, remove automatically applied UTC timezone, and convert to US/Eastern time.
    dti = pd.to_datetime(df['timestamp_local']).dt.tz_localize(None).dt.tz_localize('US/Eastern')
    df = df.assign(timestamp_local=dti)

    #order by timestamp asc instead of desc
    df = df.sort_values(by=['timestamp'])

    #drop duplicates if they appear in set
    df = df.drop_duplicates(ignore_index=True)

    #munging dataset?

    #values <0 are set to NaN, windspeeds or concentrations or both? what about these ones: "pm25", "pm10"?
    cols = ["o3", "co", "co2", "no2"] #, "bin0", "pm1"] #TODO need raw data for these
    for c in cols:
        print(c, len(df[df[c] < 0]), len(df[(df[c] > CUTOFF)]))
        df.loc[df[c] < 0, c] = np.nan
        #values over 300 are set to 0, which values are being set to 0? the concentrations or the wind speeds? should it be zero or NaN?
        #TODO all the co2 values were over 300 here
        df.loc[df[c] > CUTOFF, c] = 0
    #TODO: baseline correction function? check out the online documentation about it, is it necessary for revere?

    return df

def check_df(df):
    """
    sanity check that all values for each sensor are within reasonable range
    """
    #check number of NA values, zero values, and negative values in each relevant row of the dataset
    #for these columns "co", "no" , "no2", "o3" ,  "pm1" , "co2", "no_ae", "bin0"
    cols = ["co", "no", "no2", "o3", "pm1", "co2"] #, " no_ae", "bin0"]
    sub_df = df[cols]
    zeros = (sub_df == 0).astype(int).sum(axis=0)
    negs = (sub_df < 0).astype(int).sum(axis=0)
    nans = (sub_df == np.nan).astype(int).sum(axis=0)

    #calculate mean, 25th and 75th percentile for each relevant column
    mean = sub_df.mean(axis=0)
    quantile = sub_df.quantile([0.25, 0.75])

    #print results
    print("---- number of rows that were 0 for each column ----")
    print(zeros)
    print("---- number of rows that were LESS THAN 0 for each column ----")
    print(negs)
    print("---- number of rows that were NaN for each column ----")
    print(nans)

    print("---- mean of each column ----")
    print(mean)
    print("---- 25th and 75th percentile for each column ----")
    print(quantile)

    return zeros, negs, nans, mean, quantile

def flags(df):
    """
    extract flagged values and plot??
    """
    flag_cutoffs = {
        "co2": 2000,
        "no2": 1000,
        "no": 300
    }
    #generate df of values where a given variable was flagged ie
    df_co2 = df[df.co2 >= 2000]
    #make sure that the interval lasts at least 2 minutes for continuous section of data, maybe we maintain the indexes?
    #adds 5-10 minutes before and after the spikes
    #any remaining interval gets plotted... there's a lot of these, how can it be improved?
    pass

def plot_dfs(dfs):
    """
    Create some plot for a dataframe....
    TODO
    """
    pass


######################### UTILITY FUNCTIONS #########################
def save_files(df, path):
    """
    Save a cleaned Dataframe as a pickle file for now
    TODO: if the files become too large we can switch over to using Apache feather files which have better compression for dfs
    specifically but don't play well with edits (i.e. better to open->read->make edits in Python->create+store in a new file)
    """
    with open(path, 'wb') as f:
        pickle.dump(df, f)

def load_df(path):
    """
    Load a stored Dataframe
    """
    with open(path, 'rb') as f:
        df = pickle.load(f)
    return df

def import_data(path):
    return pd.read_csv(path)

def read_token():
    with open(TOKEN_PATH, 'r') as f:
        token = f.read()
    return token


if __name__ == "__main__":
    should_pull_raw = False
    start_time = datetime(2021, 3, 3)
    end_time = datetime(2021, 3, 5)

    #initialize QuantAQ API client
    client = quantaq.QuantAQAPIClient(api_key=read_token())

    #request data from QuantAQ for a given start-end date range
    data = request_data(sensor, start_time, end_time)
    if should_pull_raw:
        data_raw = request_data(sensor, start_time, end_time, raw=True)
        print(data_raw)
        data = df.join(df_raw)

    #sanity check the df before cleaning
    check_df(data)

    #clean the df based on Alli's procedure
    df = cleaning_final(data)

    #store the cleaned df
    save_files(df, "qaq_cleaned_data/test.pckl")

    #pick and view segments of dataset that have pollutant spikes
    flagged_dfs = flags(df)
    plot_dfs(flagged_dfs)

    #having issues grabbing data via API so we just import a local file for now
    # df = import_data("test.csv")
    print(df.head())
    print(df.dtypes)
    print(len(data))