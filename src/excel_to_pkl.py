# import xlrd
# import 

# data = xlrd.open_workbook('test.xlsx')
# table = data.sheets()[0]

# head = table.row_values(0)

import xlrd
import pandas as pd
import pickle as pkl

fp = xlrd.open_workbook('NREL_Solar_Dataset.xlsx')
table = fp.sheets()[0]

data = dict()
for i in range(table.ncols):
	col = table.col_values(i)
	head = col.pop(0)
	data[head] = col

df = pd.DataFrame(data)

output = open('solar_data.pkl', 'wb')
pkl.dump(df,output)