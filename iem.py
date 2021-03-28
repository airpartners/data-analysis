"""
Author: Hwei-Shin Harriman
Code to automatically pull meteorology data from IEM website based on following specifications: 
    * BOS station
    * Wind Direction
    * Wind Speed [mph]
    * Selected date range (dataset is not inclusive of last date):
    * America/New_York timezone
    * csv 
    * no latitude/ longitude vectors 
    * represent missing data with blank string
    * denote trace with blank string
Specifications based on https://github.com/scott-hersey/EB_AQ_Network/blob/master/initial_analysis_walkthrough.Rmd
"""
import json
import time
import datetime
import pandas as pd
from urllib import parse, request
from io import StringIO

# Number of attempts to download data
MAX_ATTEMPTS = 6
SERVICE = "http://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"

# based on specifications these should not change unless we decide to change sensor networks
DEFAULT_PARAMS = {
    "station": "BOS",   #sensor station
    "data": "drct",     #wind direction in deg from north. we also want wind speed, see ADDTL_PARAMS_STR
    "tz": "utc",  #timezone
    "format": "onlycomma",   #csv
    "latlon": "no",     #no latitude/longitude
    "elev": "no",       #no elevation
    "missing": "empty", #missing data values = empty string
    "trace": "empty",   #missing data values = empty string
    "direct": "no",     # i'm not sure what this means, it is included when downloading from webpage
    "report_type": 1    #IEM encoded code for the report type
}
# IEM query string pattern adds new key, value pair for each data column requested, i.e. 2 data cols = data=sped&data=drct...
# python cannot have more than 1 identical key in a dictionary, so we have these hardcoded. Same issue for report_type, unfortunately
ADDTL_PARAMS_STR = "&data=sped&report_type=2"

def download_data(uri):
    """Fetch the data from the IEM
    The IEM download service has some protections in place to keep the number
    of inbound requests in check.  This function implements a backoff 
    to keep individual downloads from erroring.
    Args:
      uri (string): URL to fetch
    Returns:
      string data
    """
    attempt = 0
    while attempt < MAX_ATTEMPTS:
        try:
            data = request.urlopen(uri, timeout=300).read().decode("utf-8")
            if data is not None and not data.startswith("ERROR"):
                return data
        except Exception as exp:
            print("download_data(%s) failed with %s" % (uri, exp))
            time.sleep(5)
        attempt += 1

    print("Exhausted attempts to download, returning empty data")
    return ""

def make_request_uri(start, end):
    """
    Builds request URI for meteorology data
    :param start: datetime object for start time of data
    :param end: datetime object for end time of data
    """
    params = {
        "year1": start.year,
        "month1": start.month,
        "day1": start.day,
        "year2": end.year,
        "month2": end.month,
        "day2": end.day,
    }
    #combine variable parameters with constant parameters into 1 dict
    params.update(DEFAULT_PARAMS)

    #build/return full uri
    return SERVICE + parse.urlencode(params) + ADDTL_PARAMS_STR

def fetch_data(start, end):
    """
    Makes a pandas DataFrame from Boston Logan sensor from IEM website.

    :param start: (datetime) beginning of time range to pull data for
    :param end: (datetime) end of time range to pull data for
    :returns: pandas Dataframe containing results
    """
    uri = make_request_uri(start, end)
    # fetches data from IEM, data is a formatted string with "," separators
    data = download_data(uri)
    #convert string into a stream so that it can be "read" into a pandas dataframe
    data_stream = StringIO(data)
    #convert to dataframe
    df = pd.read_csv(data_stream, sep=",")
    return df


if __name__ == "__main__":
    print(fetch_data(datetime.datetime(2012, 8, 1), datetime.datetime(2012, 9, 1)))