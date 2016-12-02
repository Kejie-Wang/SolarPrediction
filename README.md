# SolarPrediction

##Introduction
Use netural network to prediction the solar irradiance irradiation

## Dependency

### Platform
Ubuntu 14.04

###Library

- **Python2.7:** [**Anaconda4.2.0**](https://www.continuum.io/downloads)
- **Tensorflow-0.11:** Use [**Anaconda Installation**](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#anaconda-installation)
- **numpy, matplotlib, PIL, xlrd, urllib, urllib2:**
	- **conda install:**`conda install numpy, matplotlib, PIL, xlrd`
	- **pip install:** `pip install numpy, matplotlib, PIL, xlrd`

## How To Run
Before run this project, make sure that you have installed all dependecy and you should use the python script to download the dataset by yourself since the dataset is too large that I have not upload it into the Github. 

Download the dataset (*this may cost severals hours to get the data and please wait patiently*):
- Download the nrel irradiance and meteorological data from the [**SSRL_BMS**](https://www.nrel.gov/midc/srrl_bms/)
    ***`# python dataset/NREL_SSRL_BMS_IRANDMETE/ssrl_bms_historical_data_spider.py`***
- Download the ssrl sky image from the [**skycam**](https://www.nrel.gov/midc/skycam).  
***`# python dataset/NREL_SSRL_BMS_SKY_CAM/ssrl_sky_image_spider_multi_thread.py`***

Data preprocess
- NREL solar and temperature data pre-process  
    ***`# python dataset/NREL_SOLAR_DATASET/nrel_solar_preprecess.py`***
  
- Run the model  
 ***`# python src/solar_prediction_model.py`***

## Dataset
This project is focus on the solar irradiance prediction according to the irradiance data, meteorological data and sky camera data from the [NREL Solar Radiation Research Laboratory](https://www.nrel.gov/midc/srrl_bms/).
The dataset consists of three parts:
- **Irradiance Data and Meteorological Data**
The dataset can be download from the [SSRL_BMS Historical Monthly Data](https://www.nrel.gov/midc/srrl_bms/historical/). The field of the dataset are different since the system added some equiments in some year. You can check the field id of each year from the [data field definition](https://www.nrel.gov/midc/srrl_bms/historical/qa_fd.html). I save the field into csv files by year in the [dataset/NREL_SSRL_BMS_IRANDMETE/field_id/](https://github.com/JackWang822/SolarPrediction/tree/master/dataset/NREL_SSRL_BMS_IRANDMETE/field_id) and write a python script to generate the common irradiance data and meteorological data . since it is very  troublesome to handle this field. You can easily ignore the field and just focus the common id since 2006.
***Note: The dataset that downloaded from the website may exist some problems such as 2008, 200908, 201007 and so on. The problems is the column is not same in the file and it will be a problem when I use the genfromtxt function in the numpy to read the csv file. The solution to this is that I open the txt file in the excel and resave it as csv file and replace it.***
