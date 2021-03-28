# Data Analysis
This repository contains data and code associated with analysis of data gathered from the network of sensor boxes in East Boston.

## Contents
Current contents of this repository include two notebooks associated with some exploration over Spring 2020 ([looking at the raw data](https://github.com/airpartners/data-analysis/blob/master/SP19-raw-data-analysis-test.ipynb) and [doing some pandas processing](https://github.com/airpartners/data-analysis/blob/master/SP19-data-analysis-test.ipynb)) and some [preliminary AQI computations and analysis](https://github.com/airpartners/data-analysis/blob/master/aqi-analysis.ipynb) [associated with an assumpion test](https://docs.google.com/document/d/1EtGzySFGZ5PVPhCsywW77AdkwYG_axO-Zp_cfnLsTVg/edit?usp=sharing) conducted in Fall 2020.

## Opening the Jupyter Notebook (.ipynb)
You need to install several Python packages to use the jupyter notebook.
```
pip3 install --upgrade pip
pip3 install --upgrade ipython jupyter pandas numpy matplotlib
```

After those commands complete, `cd` into the data-analysis repo and type `jupyter notebook`. The directory should open in your browser and then you can open the notebook by simply clicking on the file name.
