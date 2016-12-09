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
uf.CombineTheTwoChannels(file)

#!h5dump -H $file
f = h5py.File(file, 'r')

i1_indexes = f['LowPassSegmentation/i1/CommonIndex']

print(i1_indexes[:])