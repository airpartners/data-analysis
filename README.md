# Data Analysis
Various tools for QuantAQ sensor data analysis. See `notebooks/` for the older version of this repo which contains jupyter notebooks for visual data analysis.

# Pipeline
This pipeline automates the data pulling and cleaning process for grabbing data from the QuantAQ network of sensors.
### Prerequisites
1. These instructions assume you are running at least Python 3.8
2. Run `pip install -r requirements.txt` to install the Python dependencies.
3. Get an API key for QuantAQ. You can ask Scott Hersey for his API key (there have been issues in the past if you are not an admin for the sensor network you're trying to pull data from). Copy the key into a file called `token.txt` in the root of this repository.

## Usage
Check out the `demo.py` script for example usage!

By default, cleaned data results are returned as a pandas `DataFrame` and also stored as a `pickle` file.

### Pulling from QuantAQ via Rest API
**Note that requesting data from QuantAQ is slow! It takes on the order of 2-3 minutes per sensor, per day**

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

# Visualizing Plots with OpenAir and `rpy2`
## Prerequisites - R, OpenAir, `rpy2` Installations
If you want to visualize dataframe results with the R package, OpenAir, you will need to install the necessary R dependencies. To be specific, any methods found in the `dataviz.py` file require R. This involves installing R, installing OpenAir (and its dependencies).

### **Mac OS Instructions**
1. R installation: [link](https://cran.r-project.org/doc/manuals/r-release/R-admin.html)
2. OpenAir installation: Open the R application. In the taskbar, select `Packages & Data` dropdown menu, and click on  `Package Installer`. This should open up a window. From there, make sure that `CRAN binaries` is selected from the first dropdown menu. Type "openair" into the search bar. It might prompt you to select a region. I picked `US [IW]`, then pressed `Get List`. I selected the first result. At the bottom, make sure that `Install Dependencies` is checked, then click `Install Selected`. **NOTE: I did these steps in a much more jank way, so I'd be happy to know if someone has success replicating this process.**
3. Run `export R_HOME=/Library/Frameworks/R.framework/Resources`
4. Run `pip install rpy2` to install the `rpy2` package.

### **Ubuntu 20.04 Instructions**
1. Install gdebi:
```[bash]
sudo apt update
sudo apt -y install r-base gdebi-core
```
2. Get the most recent RStudio version - I selected the `.deb` file for Ubuntu 18 - from [here](https://www.rstudio.com/products/rstudio/download/#download).

3. Run `sudo gdebi rstudio-1.2.5019-amd64.deb`
4. add `deb http://security.ubuntu.com/ubuntu xenial-security main` to `/etc/apt/sources.list`
5. `sudo apt update`
`sudo apt install libssl1.0.0,1
6. Deleted the line I added to sources.list for good measure, probably not necessary/will mess things up next time I update but that's a choice I made
7. Open the RStudio application
8. Download OpenAir package and dependencies from RStudio: Tools > Install Packages > Type "openair" into the search bar > make sure "install dependencies:" is checked > Install
9. Run `export R_HOME=/usr/lib/R`
10. Run `pip install rpy2` to install the `rpy2` package.

### **Windows Instructions**
**NOTE: I haven't been able to test these instructions myself, so definitely feel free to update these instructions if you encounter any difficulty**
* Install Windows RStudio:
1. Get the most recent RStudio version from [here](https://www.rstudio.com/products/rstudio/download/#download).
2. Run the installer
3. Open RStudio application
4. Download OpenAir package and dependencies from RStudio: Tools > Install Packages > Type "openair" into the search bar > make sure "install dependencies:" is checked > Install
5. You might need to export your `R_HOME` to run the Python code
6. Run `pip install rpy2` to install the `rpy2` package

## Usage
Refer to the `demo_viz.py` script for example usage!

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