import numpy as np
import operator

field_id_0003 = np.loadtxt('200011_200312_id.csv', dtype='str', delimiter=',',comments=None)
field_id_0404 = np.loadtxt('200401_200407_id.csv', dtype='str', delimiter=',',comments=None)
field_id_0405 = np.loadtxt('200408_200512_id.csv', dtype='str', delimiter=',',comments=None)
field_id_0608 = np.loadtxt('200601_200806_id.csv', dtype='str', delimiter=',',comments=None)
field_id_0812 = np.loadtxt('200807_201205_id.csv', dtype='str', delimiter=',',comments=None)
field_id_1214 = np.loadtxt('201206_201412_id.csv', dtype='str', delimiter=',',comments=None)
field_id_1516 = np.loadtxt('201501_201610_id.csv', dtype='str', delimiter=',',comments=None)

field_id_0003_dict = dict(field_id_0003[:,[2,0]])
field_id_0404_dict = dict(field_id_0404[:,[2,0]])
field_id_0405_dict = dict(field_id_0405[:,[2,0]])
field_id_0608_dict = dict(field_id_0608[:,[2,0]])
field_id_0812_dict = dict(field_id_0812[:,[2,0]])
field_id_1214_dict = dict(field_id_1214[:,[2,0]])
field_id_1516_dict = dict(field_id_1516[:,[2,0]])

meteorological_id = np.loadtxt('meteorological_id.csv', dtype='str', delimiter=',', comments=None)
irradiance_id = np.loadtxt('irradiance_id.csv', dtype='str', delimiter=',', comments=None)

field_id_common_label = reduce(np.intersect1d, (field_id_0003[:,2], \
                                                field_id_0404[:,2], \
                                                field_id_0405[:,2], \
                                                field_id_0608[:,2], \
                                                field_id_0812[:,2], \
                                                field_id_1214[:,2], \
                                                field_id_1516[:,2]))

#get all common label the index in the field
#assert all common filed id are with the same index in the filed id
field_id_common_mete = dict()
field_id_common_irra = dict()

for id in field_id_common_label:
    if id in meteorological_id:
        field_id_common_mete[id] = field_id_0608_dict[id]
    elif id in irradiance_id:
        field_id_common_irra[id] = field_id_0608_dict[id]
    else:
        print id
    assert field_id_0003_dict[id] == field_id_0404_dict[id] and \
            field_id_0404_dict[id] == field_id_0405_dict[id] and \
            field_id_0405_dict[id] == field_id_0608_dict[id] and \
            field_id_0608_dict[id] == field_id_0812_dict[id] and \
            field_id_0812_dict[id] == field_id_1214_dict[id] and \
            field_id_1214_dict[id] == field_id_1516_dict[id]

np.savetxt('meteorological_common_id.csv',  np.array(field_id_common_mete.items()), fmt='%s',  delimiter=',',comments=None)
np.savetxt('irradiance_common_id.csv',  np.array(field_id_common_irra.items()), fmt='%s',  delimiter=',',comments=None)
