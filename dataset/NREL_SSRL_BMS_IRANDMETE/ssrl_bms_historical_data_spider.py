import urllib2
import os

def get_file_name(year, month):
	return str(year).rjust(4, '0') + str(month).rjust(2, '0') + ".zip"

def download_file(url, file_path):
	f = urllib2.urlopen(url)
	with open(file_path, 'wb') as fp:
		data = f.read()
		fp.write(data)
		fp.close()

ys, ms = 2006, 1
ye, me = 2016, 7

url = 'https://www.nrel.gov/midc/srrl_bms/historical/data/'
file_path = "./raw_data/"

if not os.path.exists(file_path):
	os.mkdir(file_path)
if not os.path.exists(file_path + str(ys)):
	os.mkdir(file_path+str(ys))
yit, mit = ys, ms
while (yit < ye) or (yit==ye and mit<=me):
	file_name = get_file_name(yit, mit)
	download_file(url+file_name, file_path+str(yit).rjust(4, '0')+'/'+file_name)
	mit += 1
	if mit > 12:
		mit = 1
		yit += 1
		if not os.path.exists(file_path + str(yit).rjust(4, '0')):
			os.mkdir(file_path + str(yit).rjust(4, '0'))
