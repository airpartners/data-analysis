# Data Analysis
Various tools for QuantAQ sensor data analysis. See `notebooks/` for the older version of this repo which contains jupyter notebooks for visual data analysis.

## Pipeline
This pipeline automates the data pulling and cleaning process for grabbing data from the QuantAQ network of sensors.
### Prerequisites
1. These instructions assume you are running at least Python 3.8
2. Run `pip install -r requirements.txt` to install the Python dependencies.
3. Get an API key for QuantAQ. You can ask Scott Hersey for an API key. Copy the key into a file called `token.txt` in the root of this repository.
4. If you want to visualize things with the R package, OpenAir, you will need to install the necessary R dependencies. This involves installing R, installing OpenAir (and its dependencies).
    * R installation: [link](https://cran.r-project.org/doc/manuals/r-release/R-admin.html)
    * OpenAir installation: Open the R application. In the taskbar, select `Packages & Data` dropdown menu, and click on  `Package Installer`. This should open up a window. From there, make sure that `CRAN binaries` is selected from the first dropdown menu. Type "openair" into the search bar. It might prompt you to select a region. I picked `US [IW]`, then pressed `Get List`. I selected the first result. At the bottom, make sure that `Install Dependencies` is checked, then click `Install Selected`. **NOTE: I did this in a much more jank way, so I'd be happy to know if someone has success replicating this process.**

### Usage
#### Fetching Data
By default, cleaned data results are returned as a pandas `DataFrame` and also stored as a `pickle` file. As of March 28th, 2021, you might have to create a data-specific folder to store your results.

First, the imports:
```[python]
from datetime import datetime
from quantaq_pipeline import SNHandler, ModPMHandler
```
* If you are using a gas phase sensor (i.e. its serial ID starts with `SN`):
```[python]
#initalize handler that will pull data from the "your-sn-sensor-id" sensor
handler = SNHandler(sensor_id="your-sn-sensor-id")

#define start and end date for sensor collection from Mar 5th-9th, 2021
start_date, end_date = datetime(2021, 3, 5), datetime(2021, 3, 9)

#define a recognizable prefix to find the stored results
save_name = "my-sn-sensor-file"

#run the pipeline, gives you a pandas dataframe to play with!
df = handler.main(start_date, end_date, save_name)
```
* If you are using a modular PM sensor (i.e. its serial ID starts with `MOD-PM`) everything is the same as the SN sensor except you intiailize a different class:
```[python]
handler = ModPMHandler(sensor_id="your-mod-pm-sensor-id")
#... initialize a start/end date and a save_name
df = handler.main(start_date, end_date, save_name)
```

#### Visualization with Openair
Once you have your data in a pandas `DataFrame`, you can call some visualization functions on it. This is a basic example, refer to the docstrings in the `dataviz.py` file for more detailed information.
```[python]
from dataviz import OpenAirPlots

#grab data from an SN sensor like we did above...
handler = SNHandler(sensor_id="your-sn-sensor-id")
df = handler.main(start_date, end_date, save_name)

#intialize openair plotting class
plt = OpenAirPlots()

#create time variation plots
plt.time_variation(df, save_name, handler.data_cols)
#create polar plots
plt.polar_plot(df, save_name, handler.data_cols)
```