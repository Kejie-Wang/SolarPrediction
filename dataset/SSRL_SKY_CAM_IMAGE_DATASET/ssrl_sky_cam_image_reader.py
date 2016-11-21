import os
from datetime import datetime, date, timedelta
import Image

def isvalid(year, month, day):
	"""Check if a date is valid
	if it is valid, return True otherwise False
	e.g. 20150229 is not a valid day and return False
		 20150101 is a valid day and return True
	"""
	try:
		newdate = datetime(year, month, day)
		return True
	except ValueError:
		return False

def get_file_path(year, month, day, hour):
	DIR_PATH = "SSRL_SKY_CAM_IMAGE"

	ystr = str(year) + rjust(4, '0')
	mstr = str(month) + rjust(2, '0')
	dstr = str(day) + rjust(2, '0')
	hstr = str(hour) + rjust(2, 0)

	return DIR_PATH + "/" + ystr + "/" + mstr + "/" + dstr + "/" +  
		+ ystr + mstr + dstr + hstr + "00.jpg"

def get_an_image(year, month, day, hour):
	# dt = datetime.strptime(time)
	# year, month, day, hour = dt.year, dt.month, dt.day, dt.hour
	file_path = get_file_path(year, month, day, hour)
	if os.path.isfile(file_path):
		im = Image.open(file_path)
	else:
		file_path_before = get_file_path(year, month, day, hour-1)
		file_path_after = get_file_path(year, month, day, hour+1)
		im1 = Image.open(file_path_before)
		im2 = Image.open(file_path_after)
		im = Image.blend(im1, im2, 0.5)
	return list(im.getdata())

def get_images(start, end):
	"""Get a list of images from the dataset for the given start and end date
		start: the start date of string type(YYYYMMDD)
		end: the end date of string type(YYYYMMDD)
	"""
	dts = datetime.strptime(start)
	ys, ms, ds= dts.year, dts.month, dts.day
	dte = datetime.strptime(time)
	ye, me, de= dte.year, dte.month, dte.day

	images = dict()
	it = date(ys, ms, ds)
	end = date(ye, me, de)
	while it <= end:
		for hour in range(5, 20):
			dt = datetime(it.year, it.month, it.day, hour)
			images[dt] = get_an_image(it.year, it.month, it.day, hour)
		it += timedelta(days = 1)
	
	return images