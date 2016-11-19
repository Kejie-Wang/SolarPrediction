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
- Download the ssrl sky image from the [**skycam**](https://www.nrel.gov/midc/skycam).  
***`# python dataset/SSRL_SKY_CAM_IMAGE_DATASET/ssrl_sky_image_spider_multi_thread.py`***
- NREL solar and temperature data pre-process  
  
***`# python dataset/NREL_SOLAR_DATASET/nrel_solar_preprecess.py`***
  
- Run the model  
 ***`# python src/solar_prediction_model.py`***
