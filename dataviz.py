"""
Author: Hwei-Shin Harriman
Project: Air Partners

Functions to render various air quality graphs
"""
import os
from pathlib import Path
#TODO this path only works for Mac, make OS specific (i.e. on Ubuntu the R_HOME is at "/usr/lib/R")
if not os.environ.get("R_HOME"):
    os.environ['R_HOME'] = "/Library/Frameworks/R.framework/Resources"
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro

def plot(df, handler, sensor_id, save_prefix=None, smoothed=True, cols=None):
    """
    Create diurnal and polar plots for relevant columns in dataframe.

    :param df: (pd.DataFrame) dataframe containing cleaned AQ data
    :param handler: (either SNHandler or ModPMHandler from quantaq_pipeline) the data handler for the SN or MOD-PM sensor
    :param smoothed: (optional bool) True if unrealistically high values were removed from df
    :param cols: (optional list) if defined, determines the columns of df to include in the plots
    :returns: null but saves OpenAir figures as .pngs locally.
    """
    plot_path = f"img/{sensor_id}"
    #make image directories for this sensor if they do not exist already
    Path(plot_path).mkdir(parents=True, exist_ok=True)
    save_prefix = save_prefix if save_prefix else handler.get_save_name(smoothed=smoothed)
    prefix = os.path.join(plot_path, save_prefix)

    cols = cols if cols else handler.data_cols
    #make sure that only columns that have data will be plotted
    non_null_cols = []
    for c in cols:
        if not df[c].isnull().all():
            non_null_cols.append(c)

    #generate diurnal and polar plots for the non_null columns
    plt = OpenAirPlots()
    plt.time_variation(df, prefix, non_null_cols)
    plt.polar_plot(df, prefix, non_null_cols)

class OpenAirPlots:
    """
    Wrapper class to call OpenAir functions from R.
    """
    def __init__(self):
        self.openair = importr('openair')
        self.grdevices = importr('grDevices')

    def _convert_df(self, df):
        """
        Convert a pandas dataframe to an OpenAir-compatible rpy2 dataframe

        :param df: (pd.DataFrame) dataframe to convert
        :returns: converted dataframe with specific columns renamed to fit requirements of OpenAir
        """
        pandas2ri.activate()
        #for openair we need to have a few specific column names
        df = df.rename(columns={"timestamp_local": "date", "wind_speed": "ws", "wind_dir": "wd"})
        # To R DataFrame
        r_df = ro.conversion.py2rpy(df)
        return r_df

    def time_variation(self, df, file_prefix, pollutants, width=1200, height=700):
        """
        Saves a .png diurnal plot of pollutants over time

        :param df: (pd.DataFrame) cleaned dataframe containing pollutant sensor data
        :file_prefix: (str) unique identifier to save the plot
        :pollutants: (list of str) array of columns corresponding to pollutants to be included in plot
        :param width: (optional int) width in px of the figure
        :param height: (optional int) height in px of the figure
        :returns: null
        """
        #convert pandas df to rpy2 df
        r_df = self._convert_df(df)

        #build string representation of pollutants to pass into openair
        pols = ""
        for p in pollutants:
            pols += f'"{p}", '
        pols = pols.rstrip(", ")
        pols = f"c({pols})"
        print(pols)

        #create diurnal plot, may take a few seconds to complete
        self.grdevices.png(f"{file_prefix}_diurnal.png", width=width, height=height)
        ro.r.timeVariation(r_df, pollutant = ro.r(pols), normalise = True, main = "Normalized Group of Pollutants Diurnal Profile")
    
    def polar_plot(self, df, file_prefix, pollutants, width=700, height=700):
        """
        Saves a .png polar plot for each pollutant. The polar plot visualizes the concentration and direction of origin
        of a given pollutant relative to the location of the sensor box.

        :param df: (pd.DataFrame) cleaned dataframe containing pollutant sensor data
        :param file_prefix: (str) unique identifier to save the plot
        :param pollutants: (list of str) array of columns corresponding to pollutants. each pollutant will get its own plot
        :param width: (optional int) width in px of the figure
        :param height: (optional int) height in px of the figure
        :returns: null
        """
        #convert dataframe to rpy2 df
        r_df = self._convert_df(df)

        #save a polar plot for each pollutant, may take several seconds to complete
        for p in pollutants:
            self.grdevices.png(f"{file_prefix}_polar_{p}.png", width=width, height=height)
            ro.r.polarPlot(r_df, pollutant = p, main = f"{p.upper()} Polar Plot")