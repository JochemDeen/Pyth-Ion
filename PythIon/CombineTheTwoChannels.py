import UsefulFunctions as uf
import pyqtgraph as pg
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import scipy
import numpy as np
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import h5py

file='/Users/migraf/Desktop/Temp/17B_10mMCis100mMtransKCl_80mer_5_OriginalDB.hdf5'
f = h5py.File(file, 'a')

i1 = f['LowPassSegmentation/i1/']
i2 = f['LowPassSegmentation/i2/']

#Common Events
#Take Longer
CommonEventsi1Index=np.array([], dtype=np.uint64)
CommonEventsi2Index=np.array([], dtype=np.uint64)
DelayLimit=100

for i in range(len(i1['StartPoints'])):
    for j in range(len(i2['StartPoints'])):
        if np.absolute(i1['StartPoints'][i]-i2['StartPoints'][j]) < DelayLimit:
            CommonEventsi1Index=np.append(CommonEventsi1Index, i)
            CommonEventsi2Index=np.append(CommonEventsi2Index, j)
#Only i1
Onlyi1Indexes=np.delete(range(len(i1['StartPoints'])), CommonEventsi1Index)
#Only i2
Onlyi2Indexes=np.delete(range(len(i2['StartPoints'])), CommonEventsi2Index)

e = "CommonIndex" in i1
if e:
    del i1['CommonIndex']
    i1.create_dataset('CommonIndex', CommonEventsi1Index)
    del i2['CommonIndex']
    i2.create_dataset('CommonIndex', CommonEventsi2Index)
    del i1['OnlyIndex']
    i1.create_dataset('OnlyIndex', Onlyi1Indexes)
    del i2['OnlyIndex']
    i2.create_dataset('OnlyIndex', Onlyi2Indexes)
else:
    i1.create_dataset('CommonIndex', CommonEventsi1Index)
    i2.create_dataset('CommonIndex', CommonEventsi2Index)
    i1.create_dataset('OnlyIndex', Onlyi1Indexes)
    i2.create_dataset('OnlyIndex', Onlyi2Indexes)

print('Common i1: ' + str(CommonEventsi1Index))
print('Common i2: ' + str(CommonEventsi2Index))
print('Only i2: ' + str(Onlyi2Indexes))
print('Only i1: ' + str(Onlyi1Indexes))