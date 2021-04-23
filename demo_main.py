from dataviz import plot, OpenAirPlots
import quantaq_pipeline as qp

"""
Sample script showing the functionality of the pipeline + creating OpenAir visualizations. Make sure that you have
followed the setup instructions in the README before running this code. If you have set up your R environment and project
environment correctly then this code should run without errors.

Author: Hwei-Shin Harriman
Project: Air Partners
"""

#define constants for SN sensor
sn_id = "SN000-046"
final_sn, raw_sn = "test_csvs/SN046_final.csv", "test_csvs/SN046_raw.csv"

#constants for MOD-PM sensor
mod_id = "MOD-PM-00049"
final_mod, raw_mod = "test_csvs/MOD49_final.csv", "test_csvs/MOD49_raw.csv"
mod_cols = ["pm1", "pm25", "pm10", "bin0", "opcn3_pm1", "opcn3_pm25", "opcn3_pm10", "neph_bin0", "pm1_env","pm25_env","pm10_env"]

#save plots for smoothed and un-smoothed versions of the data for the SN sensor
print(f"parsing SN data from CSV for sensor ID: {sn_id}.")
sn_handler = qp.SNHandler()
sn_df = sn_handler.from_csv(sn_id, final_sn, raw_sn)
print(f"generating diurnal and polar plots for sensor ID: {sn_id}")
#plots are saved in the ./img/<sn_id> folder
plot(sn_df, sn_handler, sn_id)      #NOTE: plotting may take a couple of minutes

sn_df = sn_handler.from_csv(sn_id, final_sn, raw_sn, smoothed=False)
plot(sn_df, sn_handler, sn_id, smoothed=False)

#do the same for the MOD-PM sensor
print(f"parsing MOD-PM data from CSV for sensor ID: {mod_id}")
mod_handler = qp.ModPMHandler()
mod_df = mod_handler.from_csv(mod_id, final_mod, raw_mod)
print(f"generating diurnal and polar plots for sensor ID: {mod_id}")
plot(mod_df, mod_handler, mod_id, cols=mod_cols)

mod_df = mod_handler.from_csv(mod_id, final_mod, raw_mod, smoothed=False)
plot(mod_df, mod_handler, mod_id, smoothed=False, cols=mod_cols)