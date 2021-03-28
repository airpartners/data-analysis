"""
Author: Hwei-Shin Harriman

Functions to render various air quality graphs
"""
import os
if not os.environ.get("R_HOME"):
    os.environ['R_HOME'] = "/Library/Frameworks/R.framework/Resources"
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro

class OpenAirPlots:
    def __init__(self):
        self.openair = importr('openair')
        self.grdevices = importr('grDevices')

    def convert_df(self, df):
        pandas2ri.activate()
        #for openair we need to have a few specific column names
        df = df.rename(columns={"timestamp": "date", "wind_speed": "ws", "wind_dir": "wd"})
        # To R DataFrame
        r_df = ro.conversion.py2rpy(df)
        return r_df

    def time_variation(self, df, file_prefix, pollutants):
        r_df = self.convert_df(df)

        #build string representation of pollutants to pass into openair
        pols = ""
        for p in pollutants:
            pols += f'"{p}", '
        pols = pols.rstrip(", ")
        pols = f"c({pols})"
        print(pols)

        self.grdevices.png(f"img/{file_prefix}_diurnal.png", width=1200, height=700)
        ro.r.timeVariation(r_df, pollutant = ro.r(pols), normalise = True, main = "Normalized Group of Pollutants Diurnal Profile")
    
    def polar_plot(self, df, file_prefix, pollutants):
        r_df = self.convert_df(df)
        for p in pollutants:
            self.grdevices.png(f"img/{file_prefix}_polar_{p}.png", width=700, height=700)
            ro.r.polarPlot(r_df, pollutant = p, main = f"{p.upper()} Polar Plot")