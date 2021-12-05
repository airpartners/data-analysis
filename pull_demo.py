import quantaq_pipeline as qp
from datetime import datetime

"""
Sample script for pulling and storing data from the Quant AQ pipeline. Make sure that you have followed
the setup instructions for the Pipeline in the README before you run this code. If everything has been
set up properly you should be able to run `python pull_demo.py` without any errors.

Author: Hwei-Shin Harriman
Project: Air Partners
"""
#define constants
sn_id = "SN000-046"
mod_id = "MOD-PM-00049"

#define a start, end
start, end = datetime(2021, 10, 1), datetime(2021, 10, 10)
#define a sensor ID to pull data from
#initialize appropriate hander
sn_handler = qp.SNHandler(start_date=start, end_date=end)
#can pull dataframe from API, will return the dataframe and will also pickle the results for you to open and use later
df = sn_handler.from_api(sn_id)    #this may take several minutes!!
print(df.head())

#the same process can be used for grabbing MOD-PM sensor data:
mod_handler = qp.ModPMHandler(start_date=start, end_date=end)
df = mod_handler.from_api(mod_id) 
print(df.head())

#also, once you have cleaned a dataframe once, you can load the cleaned dataframe again:
#load the MOD-PM dataframe we just created
df = mod_handler.load_df(sensor=mod_id, start=start, end=end)
print(df.head())