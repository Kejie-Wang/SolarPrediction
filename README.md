# SolarPrediction

## Introduction
Use netural network to prediction the solar irradiance

## Dependency

### Platform
Ubuntu 14.04 with Tesla M40 (24GB memory)

###Library

- **Python2.7:** [**Anaconda4.2.0**](https://www.continuum.io/downloads)
- **Tensorflow-0.12:** Use [**Anaconda Installation**](https://www.tensorflow.org/get_started/os_setup#anaconda_installation)
- **numpy, matplotlib, PIL, xlrd, urllib, urllib2, cv2:**
  - **conda install:**`conda install numpy, matplotlib, PIL, xlrd, cv2`
  - **pip install:** `pip install numpy, matplotlib, PIL, xlrd,cv2`

## How To Run
Before run this project, make sure that you have installed all dependency and you should use the python script to download the dataset by yourself since the dataset is too large that I have not upload it into the Github.

*Note: The ssrl_bms irrandiance and meteorological dataset exists some problems if you purely download from the website, so I upload the dataset and you can ignore this step*

**Download the dataset** (*this may cost severals hours to get the data and please wait patiently*):
- Download the nrel irradiance and meteorological data from the [**SSRL_BMS**](https://www.nrel.gov/midc/srrl_bms/)  

  `# cd dataset/NREL_SSRL_BMS_IRANDMETE/`

  `# python ssrl_bms_historical_data_spider.py`

- Download the ssrl sky image from the [**skycam**](https://www.nrel.gov/midc/skycam).  

  `# cd ../NREL_SSRL_BMS_SKY_CAM/` 

  `# python ssrl_sky_image_spider_multi_thread.py`

**Data preprocess**
- NREL irradiance and meteorological data pre-process  

  `# cd dataset/NREL_SSRL_BMS_IRANDMETE/`

  `# python ir_mete_preprocess.py`

**Seperate the data**

- Generate the irradiance and meteorological train, test and validation data  

  `# cd input_data`

  `# python generate.py 0.8 0.1`

- Generate the sky cam train, test and validation data

  `# cd ../../input_data`

  `# python generate.py 0.8 0.1`

**Run the model**

- Train the model and Do prediction  

  `# cd src`

  `# python solar_prediction.py`


## Dataset

This project focuses on the solar irradiance prediction according to the irradiance data, meteorological data and sky camera data from the [NREL Solar Radiation Research Laboratory](https://www.nrel.gov/midc/srrl_bms/).
The dataset consists of three parts:
- **Irradiance Data and Meteorological Data**  
  The dataset can be download from the [SSRL_BMS Historical Monthly Data](https://www.nrel.gov/midc/srrl_bms/historical/). The field of the dataset are different since the system added some equipments in some year. You can check the field id of each year from the [data field definition](https://www.nrel.gov/midc/srrl_bms/historical/qa_fd.html). I save the field into csv files by year in the [dataset/NREL_SSRL_BMS_IRANDMETE/field](https://github.com/JackWang822/SolarPrediction/tree/master/dataset/NREL_SSRL_BMS_IRANDMETE/field) and write a python script to generate the common irradiance data and meteorological data since it is very  troublesome to handle this field. You can easily ignore the field and just focus the common id since 2006.  
  ***Note: The dataset that downloaded from the website may exist some problems such as 2008, 200908, 201007 and so on. The problems is the column is not same in the file and it will be a problem when I use the genfromtxt function in the numpy to read the csv file. The solution to this is that I open the txt file in the excel and resave it as csv file and replace it.***
- **Sky Cam Image**  

## Contact
This project is maintained by [WANG Kejie](wang_kejie@foxmail.com) and if you have some problems or find some bugs in the procedure, please send me the email.
