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

**Note that requesting data from QuantAQ is slow! It takes on the order of 2-3 minutes per sensor, per day**

By default, cleaned data results are returned as a pandas `DataFrame` and also stored as a `pickle` file.

### File Names for Saved Dataframes
Whenever you run the pipeline (either calling `from_csv()` or `from_api()` methods), the cleaned dataframe is returned by the method, and it is also saved locally as a `pickle` file. The naming scheme for these `pickle` files is as follows:
```
#for dataframes that were not smoothed.
qaq_cleaned_data/<sensor_id>/<start_year>_<start_month>_<start_day>_<end_year>_<end_month>_<end_day>.pckl

#for dataframes that were smoothed (unrealistically high sensor readings have been removed from the dataframe)
qaq_cleaned_data/<sensor_id>/<start_year>_<start_month>_<start_day>_<end_year>_<end_month>_<end_day>_smoothed.pckl
```
If saving a dataframe from the `from_api()` call, the start and end `Y_M_D` dates are determined by the `datetime` objects that were passed to the QuantAQ API call. A dataframe saved during the `from_csv()` method will have dates determined by the first and last timestamps that appear in the dataframe. Therefore, a smoothed dataframe with min/max timestamps of `March 1st, 2021` to `March 10th, 2021`, originating from the sensor `SN000-046` would have the filepath: 
```
qaq_cleaned_data/SN000-046/2021_3_1_2021_3_10_smoothed.pckl
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
Refer to the `demo_main.py` script for example usage!