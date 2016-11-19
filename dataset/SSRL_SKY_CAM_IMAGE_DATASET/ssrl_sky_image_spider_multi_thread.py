__author__ = 'WANG Kejie<wang_kejie@foxmail.com>'
__date__ = '6/10/2016'
# -*- coding:utf-8 -*-
# a simple spider for grabbing the image 

import urllib
import urllib2
import re
import os
import threading

def mkdir(path):
	if os.path.exists(path):
		#print "directory " + path + " has already existed"
		return False
	else:
		os.makedirs(path)
		return True

def saveImg(imageURL, fileName):
	u = urllib.urlopen(imageURL)
	if u.getcode() == 404:
		return False
	data = u.read()
	f = open(fileName, 'wb')
	f.write(data)
	f.close

def dayNum(year, month):
	days = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
	if month != 2:
		return days[month-1]
	else:
		if year%4==0:
			if year%100==0:
				if year%400==0:
					return 29
				else:
					return 28
			else:
				return 29
		else:
			return 28

def twoDigitsStr(num):
	if num < 10:
		return '0' + str(num)
	return str(num)

def monthlyImgSave(year, month):
	url = "https://www.nrel.gov/midc/skycam"
	mkdir('../data/'+str(year)+'/'+twoDigitsStr(month))
	d = dayNum(year, month)
	for day in range(1, d+1):
		mkdir('../data/'+str(year)+'/'+twoDigitsStr(month)+'/'+twoDigitsStr(day))
		for time in range(5, 20):
			imgURL = url + '/' + str(year) + '/thumbs/' + twoDigitsStr(month) + twoDigitsStr(day) + twoDigitsStr(time) + "00.jpg"
			path = '../data/'+str(year)+'/'+twoDigitsStr(month)+'/'+twoDigitsStr(day) + '/' + str(year) + twoDigitsStr(month) + twoDigitsStr(day) + twoDigitsStr(time) + "00.jpg"
			saveImg(imgURL, path)

threads = []
for year in range(1999, 2015):
	mkdir('../data/'+str(year))
	for month in range(1, 13):
		t = threading.Thread(target=monthlyImgSave, args=(year, month))
		threads.append(t)
print len(threads)
for t in threads:
	t.start()


