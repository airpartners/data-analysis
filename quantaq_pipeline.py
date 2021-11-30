import quantaq
from datetime import datetime, timezone, timedelta
from pathlib import Path
import os
from iem import fetch_data
import numpy as np
import pandas as pd
import pickle

"""
Author: Hwei-Shin Harriman
Project: Air Partners
Description: Pipeline to pull and clean sensor data from QuantAQ network

General SN cleaning process informed by: https://github.com/scott-hersey/EB_AQ_Network/blob/master/initial_analysis_walkthrough.Rmd
QuantAQ REST API documentation: https://quant-aq.github.io/py-quantaq/usage.html
"""

# constants
TOKEN_PATH = "token.txt"
TODAY = datetime.today()
CUTOFF = 300

class QuantAQHandler:
    """
    Class to fetch data from QuantAQ
    """
    def __init__(self, token_path):
        self.token = self._read_token(token_path)
        self.client = quantaq.QuantAQAPIClient(api_key=self.token)

    def _read_token(self, token_path):
        with open(token_path, 'r') as f:
            token = f.read()
            return token

    def request_data(self, serial_num, start_date=TODAY-timedelta(days=2), end_date=TODAY, raw=False):
        """
        Request data from QuantAQ's API.
        
        :param serial_num: (str) serial number of the sensor
        :param start_date: (optional datetime) datetime object representing beginning of date range to download data for
        :param end_date: (optional datetime) represents end of date range to download data for. Note that end_date is EXCLUSIVE of the last day,
        so end date of 2020-01-03 will return data up until 2020-01-02 at 11:59pm.
        :param raw: (optional bool) True if requesting raw data, False otherwise
        :returns: pandas Dataframe containing data
        """
        #convert datetime start, end to strings
        start = start_date.strftime("%Y-%m-%d")
        stop = end_date.strftime("%Y-%m-%d")

        s = datetime.now()
        #perform QuantAQ request
        data = self.client.data.list(sn=serial_num, start=start, stop=stop, raw=raw)
        print(f"fetching data took {datetime.now()-s} secs")

        #convert returned info to pandas df
        return pd.DataFrame(data)

class DataHandler:
    """
    Parent class containing shared utility functions for all sensor types in QuantAQ network
    """
    def __init__(self, data_cols, start, end):
        """
        :param data_cols: (list of str) containing names of columns with pollutants to be analyzed 
        :sensor_id: (str) unique ID of quantAQ sensor
        :param start: (datetime) representing UTC time for the beginning of the date range
        :param end: (datetime) representing UTC time for end of date range
        """
        self.data_cols = [col.strip("\n") for col in data_cols]
        self.start = start
        self.end = end

    def get_save_name(self, start=None, end=None, smoothed=True):
        """
        Based on the specified date range, get the name of a file to be saved or opened

        :param smoothed: (optional: boolean) True if the file contains smoothed data.
        :param start: (optional datetime) If included, overrides self.start before opening file
        :param end: (optional datetime) If included, overrides self.end before opening file
        :returns: string representation of the saved file (not including the file type suffix)
        """
        if start:
            self.start = start
        if end:
            self.end = end
        save_name = f"{self.start.year}_{self.start.month}_{self.start.day}_{self.end.year}_{self.end.month}_{self.end.day}"
        if smoothed:
            save_name += "_smoothed"
        return save_name

    def convert_timestamps(self, df):
        """
        Convert all timestamp strings to datetime objects

        :param df: (pd.DataFrame) dataframe containing sensor data
        :returns: df with timestamps converted to datetime
        """
        #timestamp is already in UTC
        df = df.assign(timestamp=pd.to_datetime(df['timestamp']))
        # timestamp_local contains local time (but expresses it in UTC, so 18:59 Eastern is expressed as 18:59 UTC)
        # need to change the timezone without altering the hour of day.
        # So, convert to datetime, remove automatically applied UTC timezone, and convert to US/Eastern time.
        dti = pd.to_datetime(df['timestamp_local']).dt.tz_localize(None).dt.tz_localize('US/Eastern')
        df = df.assign(timestamp_local=dti)

        #order by timestamp asc instead of desc
        df = df.sort_values(by=['timestamp'])
        return df

    def check_df(self, df):
        """
        visual sanity check that all values for each sensor are within reasonable range

        :param df: (pd.DataFrame) dataframe containing sensor data
        :returns: None, just prints things
        """
        #check number of NA values, zero values, and negative values in each relevant row of the dataset
        sub_df = df[self.data_cols]
        zeros = (sub_df == 0).astype(int).sum(axis=0)
        negs = (sub_df < 0).astype(int).sum(axis=0)
        nans = (sub_df == np.nan).astype(int).sum(axis=0)

        #calculate mean, 25th and 75th percentile for each relevant column
        mean = sub_df.mean(axis=0)
        quantile = sub_df.quantile([0.25, 0.75])

        #print results
        if zeros.any() > 0:
            print("---- number of rows that were 0 for each column ----")
            print(zeros)
        if negs.any() > 0:
            print("---- number of rows that were LESS THAN 0 for each column ----")
            print(negs)
        if nans.any() > 0:
            print("---- number of rows that were NaN for each column ----")
            print(nans)

        print("---- mean of each column ----")
        print(mean)
        print("---- 25th and 75th percentile for each column ----")
        print(quantile)

    def flags(self, df):
        """
        NaN any pollutant readings that are >= 3 standard deviations larger than the sensor reading
        immediately before AND after it. These can safely be considered outliers in our data set.

        :param df: (pd.DataFrame) dataframe containing sensor data
        """
        #grab std of each pollutant column
        sub_df = df[self.data_cols]
        stdev = sub_df.std(axis=0, skipna=True)
        for c in self.data_cols:
            #add 2 new columns, one with the previous pollutant sensor reading, one with the next one
            df = df.assign(prev=df[c].shift(-1))
            df = df.assign(next=df[c].shift(1))

            #calculate cutoff for every row: a difference in more than 3 std's
            threshold = df[c] - (stdev[c] * 3)

            # uncomment these for debugging print statements
            # print(f"for pollutant: {c}. spikes: ")
            # print(df.loc[(df.prev <= threshold) & (df.next <= threshold), c])
            # print(f"before replacing spikes: {df[c].isna().sum()}")

            # replace pollutant readings where current-prev AND current-next are both >= threshold
            df.loc[(df.prev <= threshold) & (df.next <= threshold), c] = np.nan
            # print(f"after replacing spikes: {df[c].isna().sum()} NaNs")   #for debugging

            #remove the temporary columns
            df = df.drop(columns=['next', 'prev'])
        return df

    def _replace_with_iem(self, df, iem_df, is_tz_aware=True):
        """
        Wind speed and wind direction from the QuantAQ sensors are unreliable so we replace them with data from
        the IEM meteorology sensors.

        :param df: (pd.DataFrame) dataframe containing sensor data
        :param iem_df: (pd.DataFrame) dataframe containing meteorology data
        :param is_tz_aware: (optional bool) True if the raw, string representations of timestamps in df are time zone-aware
        """
        #convert str representation of timestamps to datetime
        iem_df = iem_df.assign(timestamp=pd.to_datetime(iem_df['valid']))

        #IEM data is recorded once every 5 mins, quantAQ data recorded once per minute, need to fill in rows in IEM data
        # to match quantAQ. So, for every IEM timestamp, we add 4 copies of the IEM data so that the IEM and QuantAQ dataframes
        # have the same number of rows:

        #create new timestamp column that matches the timestamps for the quantAQ data
        start, end = df.timestamp.min(), df.timestamp.max()
        if not is_tz_aware:
            start, end = start.tz_localize(None), end.tz_localize(None)
        dates = pd.date_range(start=start, end=end, freq='1Min')
        #fill new empty rows with the last valid value
        iem_df['timestamp'] = iem_df['timestamp'].dt.round('1Min') # this operation sometimes adds duplicates by rounding to the same minute.
        iem_df = iem_df.set_index('timestamp')
        # delete duplicate timestamps before reindexing, or pandas complains
        iem_df = iem_df.loc[~iem_df.index.duplicated(), :]

        #reindex meteorology data to match original dataframe
        iem_df = iem_df.reindex(dates, method='pad')

        #convert timestamp index back into a column
        iem_df = iem_df.reset_index().rename(columns={"index": "timestamp"})
        #some values might be NaN due to timestamp mismatch, fill them with the next valid value
        iem_df = iem_df.fillna(method='bfill')

        #assign the new wind direction and wind speed columns to the quantAQ dataframe
        df = df.assign(wind_dir=iem_df['drct'])
        df = df.assign(wind_speed=iem_df['sped'] * (1609/3600))  #converting to m/s, 1609 meters per mile, 3600 seconds per hr
        return df

    def _cutoffs(self, df, cols=None, smoothed=True):
        """
        Remove values that are higher than a certain threshold and set all values <0 to NaN.

        :param df: (pd.DataFrame) dataframe containing sensor data
        :param cols: (optional list of str) list of dataframe columns to check for hard cutoffs,
                    default checks all self.data_cols
        :param smoothed: (optional bool) True if the values > upper threshold should be removed
        :returns: dataframe with values outside of lower and (optionally) upper bounds removed
        """
        if not cols:
            cols = self.data_cols
        for c in cols:
            if not df[c].isnull().all():    #only edit columns that are not completely empty
                #values <0 are NaN'd
                df.loc[df[c] < 0, c] = np.nan
                #values over 300 are set to 0 if smoothing
                if smoothed:
                    df.loc[df[c] > CUTOFF, c] = 0
        return df

    def save_files(self, df, sensor, smoothed=True):
        """
        Save a cleaned Dataframe as a pickle file for now
        TODO: if the files become too large we can switch over to using Apache feather files which have better compression for dfs
        specifically but don't play well with edits (i.e. better to open->read->make edits in Python->create+store in a new file)
        
        :param df: (pd.DataFrame) dataframe to save
        :param smoothed: (optional bool) True if the dataframe was smoothed
        :returns: None
        """
        #create the save path, including missing folders, if it doesn't exist yet
        folders = f"qaq_cleaned_data/{sensor}"
        Path(folders).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(folders, f"{self.get_save_name(smoothed=smoothed)}.pckl"), 'wb') as f:
            pickle.dump(df, f)

    def load_df(self, sensor, start=None, end=None, smoothed=True):
        """
        Load a stored Dataframe from a pickle file

        :param start: (optional datetime) If included, the start date of the file to open. defaults to self.start
        :param end: (optional datetime) If included, the end date of the file to open. defaults to self.end
        :param smoothed: (optional bool) True if loading a smoothed dataframe
        :returns: loaded dataframe
        """
        folders = f"qaq_cleaned_data/{sensor}"
        save_name = self.get_save_name(smoothed=smoothed, start=start, end=end)
        with open(os.path.join(folders, f"{save_name}.pckl"), 'rb') as f:
            return pickle.load(f)
        
class SNHandler(DataHandler):
    """
    Handles functionality related to sensors with serial IDs that begin with 'SN' (gas phase sensors).
    """
    def __init__(self, start_date=TODAY-timedelta(days=2), end_date=TODAY):
        """
        :param start: (datetime) representing UTC time for the beginning of the date range
        :param end: (datetime) representing UTC time for end of date range
        """
        super().__init__(
            data_cols=["co", "no", "no2", "o3", "pm1", "co2", "no_ae", "bin0"], #default columns that we want to clean
            start=start_date,
            end=end_date
        )
        #define the columns we care about for the raw and final datasets
        self.final_cols = ["timestamp", "timestamp_local", "temp_box", "temp_manifold", "rh_manifold", "pressure", "noise", "solar", "wind_dir", "wind_speed", "co", "no", "no2", "o3", "pm1", "pm25", "pm10", "co2"]
        self.raw_cols = ["timestamp", "bin0", "bin1", "bin2", "bin3", "bin4", "bin5", "no_ae", "co_ae", "no2_ae"]

    def _clean_df(self, df, smoothed=True, local=False):
        """
        Cleans the dataframe from anomalies, parses timestamps, and adds wind_speed, wind_dir columns with reliable
        meteorology data from IEM.

        :param df: (pd.DataFrame) combined raw/final dataframe to clean
        :param smoothed: (optional bool) True if unrealistically large values should be removed
        :param local: (optional bool) True if the dataframe originated from a local CSV file (as opposed to being pulled via API)
        :returns: the cleaned dataframe with combined raw/final results
        """
        #sanity check the df before cleaning
        self.check_df(df)

        #convert timestamps from str->datetime
        df = self.convert_timestamps(df)

        #drop duplicates if they appear in set
        df = df.drop_duplicates(ignore_index=True)

        # remove values that are outside of valid range of values
        df = self._cutoffs(df, cols=["o3", "co", "no2", "bin0", "pm1", "no"], smoothed=smoothed)

        #reset start/end based on actual CSV values if the dataframe was local
        if local:
            self.start, self.end = df.timestamp.min().tz_localize(None), df.timestamp.max().tz_localize(None)
        #request data from IEM
        iem_df = fetch_data(self.start, self.end)

        #replace meteorology columns
        df = self._replace_with_iem(df, iem_df, is_tz_aware=False)

        #remove outliers
        df = self.flags(df)

        return df

    def from_csv(self, sensor_id, final_path, raw_path, smoothed=True):
        """
        Creates a cleaned dataframe based on locally stored raw and final .csv files. It is the caller's responsibility
        to make sure that the final_path and raw_path belong to the same sensor.
        
        :param sensor_id: (str) unique ID of the QuantAQ sensor that the csv data originated from
        :param final_path: (str) path to .csv containing the final data columns from an SN sensor
        :param raw_path: (str) path to .csv containing the raw data columns from an SN sensor
        :param smoothed: (optional bool) True if unrealistically large values should be removed
        :returns: the cleaned dataframe with combined raw/final results
        """
        #read final csv
        df_fin = pd.read_csv(final_path, sep=",")
        df_fin = df_fin[self.final_cols]

        #read raw csv
        df_raw = pd.read_csv(raw_path)
        df_raw = df_raw[self.raw_cols]

        #combine dataframes
        df = df_fin.merge(df_raw, on="timestamp")

        #clean dataframe
        df = self._clean_df(df, smoothed=smoothed, local=True)

        #store the cleaned df
        self.save_files(df, sensor_id, smoothed=smoothed)

        return df


    def from_api(self, sensor_id, smoothed=True):
        """
        Build a cleaned dataframe containing raw and final data for a QuantAQ sensor, with wind_dir/wind_speed
        replaced by the IEM meteorology sensor and outliers removed. Pulls data from the QuantAQ website over
        the network (requires internet connection). Note that pulling data this way is slow, about
        2-3 minutes per sensor per day.

        :param sensor_id: (str) unique ID of the QuantAQ sensor to pull data from
        :param smoothed: (optional bool) True if unrealistically large values should be removed
        :returns: cleaned pandas dataframe
        """
        #initialize client class for requesting data from QuantAQ
        client = QuantAQHandler(TOKEN_PATH)

        #get the final data from quantAQ, NOTE: SLOW! MAY TAKE SEVERAL MINUTES!
        print("pulling final data...")
        data = client.request_data(sensor_id, self.start, self.end)
        data = data[self.final_cols]

        #get the raw data from the same sensor, NOTE: SLOW! MAY TAKE SEVERAL MINUTES!
        print("pulling raw data...")
        data_raw = client.request_data(sensor_id, self.start, self.end, raw=True)
        data_raw = data_raw[self.raw_cols]

        #combine the raw/final dataframes
        df = data.merge(data_raw, on="timestamp")

        #clean the df
        df = self._clean_df(df, smoothed=smoothed)

        #store the cleaned df
        self.save_files(df, sensor_id, smoothed=smoothed)

        return df

class ModPMHandler(DataHandler):
    """
    Handles functionality related to modular PM sensors (sensor id's start with 'MOD-PM').
    """
    def __init__(self, start_date=TODAY-timedelta(days=2), end_date=TODAY):
        """
        :param start: (optional datetime) representing UTC time for the beginning of the date range
        :param end: (optional datetime) representing UTC time for end of date range
        """
        super().__init__(
            data_cols=[
            "neph_bin0", "neph_bin1", "neph_bin2", "neph_bin3", 
            "neph_bin4", "neph_bin5", "neph_pm1", "neph_pm10", 
            "neph_pm25", "opc_bin0", "opc_bin1", "opc_bin10",
            "opc_bin11", "opc_bin12", "opc_bin13", "opc_bin14",
            "opc_bin15", "opc_bin16", "opc_bin17", "opc_bin18",
            "opc_bin19", "opc_bin2", "opc_bin20", "opc_bin21",
            "opc_bin22", "opc_bin23", "opc_bin3", "opc_bin4",
            "opc_bin5", "opc_bin6", "opc_bin7","opc_bin8",
            "opc_bin9", "opc_pm1", "opc_pm10", "opc_pm25"
            ],  #default columns that we want to clean
            start=start_date,
            end=end_date
        )


    def _clean_mod_pm(self, df, smoothed=True, nested=False):
        """
        Flatten dataframe received from the MOD-PM sensors. This method is a helper function for data fetched via
        REST API. only data from the API have nested columns, the CSV's do not.
        
        :param df: (pd.DataFrame) dataframe containing mod-pm data
        :param smoothed: (optional bool) True if unrealistically large values should be removed
        :param nested: (optional bool) True if columns contain nested values (MOD-PM sensor data returns nested if pulled via API)
        :returns: cleaned dataframe
        """
        #replace timestamp info
        df = self.convert_timestamps(df)

        if nested:
            #create column names based on all keys within the dictionary
            neph_cols = [f"neph_{k}" for k in df['neph'][0].keys()]
            opc_cols = [f"opc_{k}" for k in df['opc'][0].keys()]

            #flatten columns that contain dictionaries
            df[neph_cols] = df.neph.apply(pd.Series)
            df[opc_cols] = df.opc.apply(pd.Series)
            df[['pressure', 'rh', 'temp']] = df.met.apply(pd.Series)

            #drop columns that contain dictionaries after flattening
            df = df.drop(['neph', 'opc', 'geo', 'met'], axis=1)

        #drop duplicate rows
        df = df.drop_duplicates(ignore_index=True)

        #visual sanity check df columns
        self.check_df(df)

        #clean spikes
        df = self.flags(df)
        df = self._cutoffs(df, smoothed=smoothed)

        return df

    def _iem(self, df, is_tz_aware=True):
        """
        Add wind direction and speed data to MOD-PM sensors by pulling from IEM website.

        :param df: (pd.DataFrame) dataframe containing mod-pm data
        :param is_tz_aware: (optional bool) True if the raw, string representations of timestamps in df are time zone-aware
        :returns: dataframe with added wind_speed, wind_dir columns
        """
        #request data from IEM
        iem_df = fetch_data(self.start, self.end)

        #add wind direction and speed to df
        #wind_dir and wind_speed columns are not included in original df so we need to add them
        df = df.assign(wind_dir=np.zeros_like(df['timestamp']))
        df = df.assign(wind_speed=np.zeros_like(df['timestamp']))
        df = self._replace_with_iem(df, iem_df, is_tz_aware=is_tz_aware)
        return df

    def from_api(self, sensor_id, smoothed=True):
        """
        Build a cleaned dataframe containing data for a MOD-PM sensor by pulling data from the QuantAQ website over
        the network (requires internet connection). Note that pulling data this way is slow, about
        2-3 minutes per sensor per day.

        :param sensor_id: (str) unique ID of the QuantAQ sensor to pull data from
        :param smoothed: (optional bool) True if unrealistically large values should be removed
        :returns: cleaned pandas dataframe
        """
        client = QuantAQHandler(TOKEN_PATH) #TODO make this not rely on a global variable token_path?
        df = client.request_data(sensor_id, self.start, self.end, raw=True)
        print(df.dtypes)
        # flatten and clean the dataframe
        df = self._clean_mod_pm(df, smoothed=smoothed, nested=True)

        #add wind direction and speed to df from iem
        df = self._iem(df)

        #store cleaned df
        self.save_files(df, sensor_id, smoothed=smoothed)

        return df

    def from_csv(self, sensor_id, final_path, raw_path, smoothed=True):
        """
        Creates a cleaned dataframe based on locally stored raw and final .csv files. It is the caller's responsibility
        to make sure that the final_path and raw_path belong to the same sensor.
        
        :param final_path: (str) path to .csv containing the final data columns from a MOD-PM sensor
        :param raw_path: (str) path to .csv containing the raw data columns from a MOD-PM sensor
        :param smoothed: (optional bool) True if unrealistically large values should be removed
        :returns: the cleaned dataframe with combined raw/final results
        """
        #local csv has a different column format than the API
        self.data_cols = [
            "bin0","bin1","bin2","bin3",
            "bin4","bin5","bin6","bin7",
            "bin8","bin9","bin10","bin11",
            "bin12","bin13","bin14","bin15",
            "bin16","bin17","bin18","bin19",
            "bin20","bin21","bin22","bin23",
            "opcn3_pm1","opcn3_pm25","opcn3_pm10","pm1_env",
            "pm25_env","pm10_env","neph_bin0","neph_bin1",
            "neph_bin2","neph_bin3","neph_bin4","neph_bin5",
            "pm1", "pm10", "pm25"
        ]
        #columns to keep from the final dataset, many of the columns are already present in the raw data
        final_cols = ["timestamp","pm1","pm25","pm10","pm1_model_id","pm25_model_id","pm10_model_id"]

        #read final csv into df
        df_fin = pd.read_csv(final_path)
        df_fin = df_fin[final_cols]

        #read raw csv into df
        df_raw = pd.read_csv(raw_path)
        #combine raw/final dataframes
        df = df_fin.merge(df_raw, on="timestamp")

        #clean dataframe
        df = self._clean_mod_pm(df, smoothed=smoothed)

        #find start and end times from the local file to inform IEM request
        self.start, self.end = df.timestamp.min().tz_localize(None), df.timestamp.max().tz_localize(None)

        #add wind direction and speed to df from iem
        df = self._iem(df, is_tz_aware=False)

        #store the cleaned df
        self.save_files(df, sensor_id, smoothed=smoothed)

        #find the start/end time of the file
        return df