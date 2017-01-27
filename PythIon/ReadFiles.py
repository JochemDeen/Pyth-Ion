import UsefulFunctions as uf
import pyqtgraph as pg
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import scipy
import numpy as np
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import h5py
import h5view

file='/Volumes/backup/2016/Michael/Axopatch/21112016/17B_10mMCis100mMtransKCl_80mer_2_OriginalDB.hdf5'
file_list=[]
file_list.append('/Volumes/backup/2016/Michael/Axopatch/21112016/17B_100mMCis1MtransKCl_80mer_7_OriginalDB.hdf5')
file_list.append('/Volumes/backup/2016/Michael/Axopatch/21112016/17B_100mMCis1MtransKCl_80mer_8_OriginalDB.hdf5')

for i in file_list:
    print(i)
    with h5view.open(i) as f:
        print(f)

#!h5dump -H $file
f = h5py.File(file_list[0], 'r')

i1_indexes = f['LowPassSegmentation/i1/CommonIndex']

print(i1_indexes[:])