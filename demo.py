import quantaq_pipeline as qp
from datetime import datetime

"""
Sample script for handling data from the Quant AQ pipeline. Make sure that you have followed
the setup instructions for the Pipeline in the README before you run this code. If everything has been
set up properly you should be able to run `python demo.py` without any errors.

Author: Hwei-Shin Harriman
Project: Air Partners
"""
#define constants
sn_id = "SN000-046"
mod_id = "MOD-PM-00049"

#define a start, end
start, end = datetime(2021, 3, 1), datetime(2021, 3, 10)
#define a sensor ID to pull data from
#initialize appropriate handler
sn_handler = qp.SNHandler(start_date=start, end_date=end)
#can pull dataframe from API, will return the dataframe and will also pickle the results for you to open and use later
#SMOOTHING: If a dataset is smoothed out in this context, then sensor measurements that are unrealistically high are filtered
#     out of the dataset. typically you want data smoothed to analyze trends, but you can set smoothed=False to see the 
#     "full picture". note that some granularity is lost by setting smoothed=True.
df = sn_handler.from_api(sn_id)    #this may take several minutes!!
print(df.head())

#the same process can be used for grabbing MOD-PM sensor data:
mod_handler = qp.ModPMHandler(start_date=start, end_date=end)
df = mod_handler.from_api(mod_id, smoothed=False) #smoothing is enabled by default, so if you want to see sensor readings that registered above their threshold, set smoothed=False
print(df.head())

#you can also use local csv files for analysis:
#need a path to a "final" csv and a "raw" csv
final_mod, raw_mod = "test_csvs/MOD49_final.csv", "test_csvs/MOD49_raw.csv"
df = mod_handler.from_csv(mod_id, final_mod, raw_mod)
print(df.head())

#just like with the API, you can explicitly set smoothed=False if you don't want smoothing
final_sn, raw_sn = "test_csvs/SN046_final.csv", "test_csvs/SN046_raw.csv"
df = sn_handler.from_csv(sn_id, final_sn, raw_sn, smoothed=False)
print(df.head())

#also, once you have cleaned a dataframe once, you can load the cleaned dataframe again:
#load the MOD-PM dataframe we just created
df = mod_handler.load_df(sensor=mod_id, start=datetime(2020, 12, 9), end=datetime(2021, 3, 31))
print(df.head())