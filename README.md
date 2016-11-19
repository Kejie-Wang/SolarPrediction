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
Before run this project, make sure that you have installed all dependecy
For the image dataset is too large, it is not in the project and you can use the python script in dataset/SSRL_SKY_CAM_IMAGE_DATASET/ssrl_sky_image_spider_multi_thread.py to spider the source image from the [**skycam**](https://www.nrel.gov/midc/skycam). This may cost several minutes to finish.
`python dataset/SSRL_SKY_CAM_IMAGE_DATASET/ssrl_sky_image_spider_multi_thread.py`
  
The nrel solar and temperature dataset is in the NREL_SOLAR_DATASET and the you must first run the pre-process python script to generate the csv file.
`python dataset/NREL_SOLAR_DATASET/nrel_solar_preprecess.py`