import quantaq
from datetime import datetime, timezone, timedelta
from pathlib import Path
import os
from iem import fetch_data
from dataviz import OpenAirPlots
import numpy as np
import pandas as pd
import pickle

"""
https://github.com/scott-hersey/EB_AQ_Network/blob/master/initial_analysis_walkthrough.Rmd
https://quant-aq.github.io/py-quantaq/usage.html
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
        :param start_date: (datetime) datetime object representing beginning of date range to download data for
        :param end_date: (datetime) represents end of date range to download data for. Note that end_date is EXCLUSIVE of the last day,
        so end date of 2020-01-03 will return data up until 2020-01-02 at 11:59pm.
        :param raw: (bool) True if requesting raw data, False otherwise
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
    def __init__(self, data_cols, sensor_id, start, end):
        """
        :param data_cols: (list of str) containing names of columns with pollutants to be analyzed 
        :sensor_id: (str) unique ID of quantAQ sensor
        :param start: (datetime) representing UTC time for the beginning of the date range
        :param end: (datetime) representing UTC time for end of date range
        """
        self.data_cols = [col.strip("\n") for col in data_cols]
        self.sensor = sensor_id
        self.start = start
        self.end = end
        self.pickle_path = f"qaq_cleaned_data/{self.sensor}"

    def _get_save_name(self, smoothed=False):
        save_name = f"{self.start.year}_{self.start.month}_{self.start.day}_{self.end.year}_{self.end.month}_{self.end.day}"
        if smoothed:
            save_name += "_smoothed"
        return save_name


    def convert_timestamps(self, df):
        """
        Convert all timestamp strings to datetime objects
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
        iem_df = iem_df.set_index('timestamp').reindex(dates, method='pad')

        # #convert timestamp index back into a column
        iem_df = iem_df.reset_index().rename(columns={"index": "timestamp"})
        #some values might be NaN due to timestamp mismatch, fill them with the next valid value
        iem_df = iem_df.fillna(method='bfill')

        #assign the new wind direction and wind speed columns to the quantAQ dataframe
        df = df.assign(wind_dir=iem_df['drct'])
        df = df.assign(wind_speed=iem_df['sped'] * (1609/3600))  #converting to m/s, 1609 meters per mile, 3600 seconds per hr
        return df

    def _cutoffs(self, df, cols=None, smoothed=False):
        """
        Remove values that are higher than a certain threshold
        """
        if not cols:
            cols = self.data_cols
        for c in cols:
            if not df[c].isnull().all():
                print(c, len(df[df[c] < 0]), len(df[(df[c] > CUTOFF)]))
                df.loc[df[c] < 0, c] = np.nan
                #values over 300 are set to 0
                if smoothed:
                    df.loc[df[c] > CUTOFF, c] = 0
        return df

    def save_files(self, df, smoothed=False):
        """
        Save a cleaned Dataframe as a pickle file for now
        TODO: if the files become too large we can switch over to using Apache feather files which have better compression for dfs
        specifically but don't play well with edits (i.e. better to open->read->make edits in Python->create+store in a new file)
        """
        Path(self.pickle_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(self.pickle_path, f"{self._get_save_name(smoothed)}.pckl"), 'wb') as f:
            pickle.dump(df, f)

    def load_df(self, smoothed=False):
        """
        Load a stored Dataframe
        """
        with open(os.path.join(self.pickle_path, f"{self._get_save_name(smoothed)}.pckl"), 'rb') as f:
            return pickle.load(f)

    def plot(self, df, smoothed=False, cols=None):
        """
        TODO
        """
        plot_path = f"img/{self.sensor}"
        #make image directories for this sensor if they do not exist already
        Path(plot_path).mkdir(parents=True, exist_ok=True)
        prefix = os.path.join(plot_path, self._get_save_name(smoothed))

        cols = cols if cols else self.data_cols
        #make sure that only columns that have data will be plotted
        non_null_cols = []
        for c in cols:
            if not df[c].isnull().all():
                non_null_cols.append(c)
        print(c)
        plt = OpenAirPlots()
        plt.time_variation(df, prefix, non_null_cols)
        plt.polar_plot(df, prefix, non_null_cols)
        
class SNHandler(DataHandler):
    """
    Handles functionality related to sensors with serial IDs that begin with 'SN' (i.e. gas phase sensors).
    """
    def __init__(self, sensor_id, start_date, end_date):
        super().__init__(
            data_cols=["co", "no", "no2", "o3", "pm1", "co2", "no_ae", "bin0"],
            sensor_id=sensor_id,
            start=start_date,
            end=end_date
        )
        self.final_cols = ["timestamp", "timestamp_local", "temp_box", "temp_manifold", "rh_manifold", "pressure", "noise", "solar", "wind_dir", "wind_speed", "co", "no", "no2", "o3", "pm1", "pm25", "pm10", "co2"]
        self.raw_cols = ["timestamp", "bin0", "bin1", "bin2", "bin3", "bin4", "bin5", "no_ae", "co_ae", "no2_ae"]

    def from_local_csv(self, final_path, raw_path, smoothed=False):
        df_fin = pd.read_csv(final_path, sep=",")
        df_fin = df_fin[self.final_cols]

        df_raw = pd.read_csv(raw_path)
        df_raw = df_raw[self.raw_cols]

        df = df_fin.merge(df_raw, on="timestamp")

        #sanity check the df before cleaning
        self.check_df(df)

        #clean the df based on Alli's procedure
        df = self.convert_timestamps(df)
        #drop duplicates if they appear in set
        df = df.drop_duplicates(ignore_index=True)

        df = self._cutoffs(df, cols=["o3", "co", "no2", "bin0", "pm1", "no"], smoothed=smoothed)

        self.start, self.end = df.timestamp.min().tz_localize(None), df.timestamp.max().tz_localize(None)
        #request data from IEM
        iem_df = fetch_data(self.start, self.end)

        #replace meteorology columns
        df = self._replace_with_iem(df, iem_df, is_tz_aware=False)

        #remove outliers
        df = self.flags(df)

        #store the cleaned df
        self.save_files(df, smoothed=smoothed)

        #find the start/end time of the file
        return df


    def main(self, open_existing=False):
        """
        Build a cleaned dataframe containing raw and final data for a QuantAQ sensor, with wind_dir/wind_speed
        replaced by the IEM meteorology sensor and outliers removed.

        :open_existing: (bool) True if the file has been pulled and cleaned already.
        :returns: cleaned pandas dataframe
        """
        if open_existing:
            try:
                return self.load_df()
            except:
                print("file does not exist for this sensor for the selected date range")

        #request data from QuantAQ for a given start-end date range
        client = QuantAQHandler(TOKEN_PATH)

        #get the final data from quantAQ
        print("pulling final data...")
        data = client.request_data(self.sensor, self.start, self.end)
        data = data[self.final_cols]

        #get the raw data from the same sensor
        print("pulling raw data...")
        data_raw = client.request_data(self.sensor, self.start, self.end, raw=True)
        data_raw = data_raw[self.raw_cols]

        data = data.merge(data_raw, on="timestamp")

        #sanity check the df before cleaning
        self.check_df(data)

        #clean the df based on Alli's procedure
        df = self._clean_quantaq(data)

        #request data from IEM
        iem_df = fetch_data(self.start, self.end)

        #replace meteorology columns
        df = self._replace_with_iem(df, iem_df)

        #remove outliers
        df = self.flags(df)

        #store the cleaned df
        self.save_files(df)

        return df

class ModPMHandler(DataHandler):
    """
    Handles functionality related to modular PM sensors (sensor id's start with 'MOD-PM').
    """
    def __init__(self, sensor_id, start_date, end_date):
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
            ],
            sensor_id=sensor_id,
            start=start_date,
            end=end_date
        )


    def _clean_mod_pm(self, df):
        """
        Flatten dataframe received from the MOD-PM sensors
        """
        #replace timestamp info
        df = self.convert_timestamps(df)
        
        #drop duplicate rows
        df = df.drop_duplicates(ignore_index=True)

        #create column names based on all keys within the dictionary
        neph_cols = [f"neph_{k}" for k in df['neph'][0].keys()]
        opc_cols = [f"opc_{k}" for k in df['opc'][0].keys()]

        #flatten columns that contain dictionaries
        df[neph_cols] = df.neph.apply(pd.Series)
        df[opc_cols] = df.opc.apply(pd.Series)
        df[['pressure', 'rh', 'temp']] = df.met.apply(pd.Series)

        #drop columns that contain dictionaries after flattening
        df = df.drop(['neph', 'opc', 'geo', 'met'], axis=1)

        #clean spikes
        df = self.flags(df)
        df = self._cutoffs(df)

        return df

    def main(self, open_existing=False):
        """
        Build a cleaned dataframe containing data for a MOD-PM sensor.

        :open_existing: (bool) True if the file has been pulled and cleaned already.
        :returns: cleaned pandas dataframe
        """
        if open_existing:
            return self.load_df()

        client = QuantAQHandler(TOKEN_PATH) #TODO make this not rely on a global variable token_path?
        df = client.request_data(self.sensor, self.start, self.end, raw=True)

        # flatten and clean the dataframe
        df = self._clean_mod_pm(df)

        #request data from IEM
        iem_df = fetch_data(self.start, self.end)

        #check for zeroes, negatives and NaNs
        self.check_df(df)

        #add wind direction and speed to df
        #wind_dir and wind_speed columns are not included in original df so we need to add them
        df = df.assign(wind_dir=np.zeros_like(df['timestamp']))
        df = df.assign(wind_speed=np.zeros_like(df['timestamp']))
        df = self._replace_with_iem(df, iem_df)
        #store cleaned df
        self.save_files(df)

        return df

    def from_local_csv(self, final_path, raw_path, smoothed=False):
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

        df_fin = pd.read_csv(final_path)
        df_fin = df_fin[final_cols]
        df_raw = pd.read_csv(raw_path)
        df = df_fin.merge(df_raw, on="timestamp")

        #sanity check the df before cleaning
        self.check_df(df)

        #replace timestamp info
        df = self.convert_timestamps(df)
        
        #drop duplicate rows
        df = df.drop_duplicates(ignore_index=True)

        #clean spikes
        df = self.flags(df)
        df = self._cutoffs(df, smoothed=smoothed)

        #find start and end times from the local file to inform IEM request
        self.start, self.end = df.timestamp.min().tz_localize(None), df.timestamp.max().tz_localize(None)
        #request data from IEM
        iem_df = fetch_data(self.start, self.end)

        #add wind direction and speed to df
        #wind_dir and wind_speed columns are not included in original df so we need to add them
        df = df.assign(wind_dir=np.zeros_like(df['timestamp']))
        df = df.assign(wind_speed=np.zeros_like(df['timestamp']))
        #replace meteorology columns
        df = self._replace_with_iem(df, iem_df, is_tz_aware=False)

        #store the cleaned df
        self.save_files(df, smoothed=smoothed)

        #find the start/end time of the file
        return df

if __name__ == "__main__":
    start_date = datetime(2021, 3, 1)
    end_date = datetime(2021, 3, 10)

    sn_handler = SNHandler(sensor_id="SN000-111", start_date=start_date, end_date=end_date)
    mod_handler = ModPMHandler(sensor_id="MOD-PM-00049", start_date=start_date, end_date=end_date)

    print("pulling SN data: ")
    smooth = True
    final, raw = "raw_data/SN111_final_new.csv", "raw_data/SN111_raw_new.csv"
    # # sn_df = sn_handler.main(open_existing=True)
    sn_df = sn_handler.from_local_csv(final, raw, smoothed=smooth)
    sn_handler.plot(sn_df, smoothed=smooth)

    smooth = False
    sn_df = sn_handler.from_local_csv(final, raw, smoothed=smooth)
    sn_handler.plot(sn_df, smoothed=smooth)

    # print("pulling MOD-PM data: ")
    # smooth = True
    # final, raw = "raw_data/MOD49_final.csv", "raw_data/MOD49_raw.csv"
    # mod_df = mod_handler.from_local_csv(final, raw, smoothed=smooth)
    # # mod_df = mod_handler.main()
    # mod_handler.plot(mod_df, smoothed=smooth, cols=["pm1", "pm25", "pm10", "bin0", "opcn3_pm1", "opcn3_pm25", "opcn3_pm10", "neph_bin0", "pm1_env","pm25_env","pm10_env"])

    # smooth = False
    # mod_df = mod_handler.from_local_csv(final, raw, smoothed=smooth)
    # # mod_df = mod_handler.main()
    # mod_handler.plot(mod_df, smoothed=smooth, cols=["pm1", "pm25", "pm10", "bin0", "opcn3_pm1", "opcn3_pm25", "opcn3_pm10", "neph_bin0", "pm1_env","pm25_env","pm10_env"])

    # do sensors in revere provide actionable data. is it valuable in interpreting the primary sources of pollutants in revere?
    #     documenting process -> link to Github -> can reference the at-a-glance page -> can you get the dataframe from R to make an at-a-glance page
    #TODO incorporate smoothing into main functions?