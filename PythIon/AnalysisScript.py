import numpy as np
#import pyqtgraph as pg
import UsefulFunctions as uf
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.ticker import NullFormatter
from matplotlib import pyplot as plt
import h5py
import os

expname = '10mMCis100mMTrans80mer_BigFileOnly'

file = '/Volumes/backup/2016/Michael/Axopatch/21112016/17B_10mMCis100mMtransKCl_80mer_2.dat'
datafile = '/Volumes/backup/2016/Michael/Axopatch/21112016/17B_10mMCis100mMtransKCl_80mer_2_OriginalDB.hdf5'

directory = (str(os.path.split(datafile)[0]) + os.sep + expname + '_SavedImages')

if not os.path.exists(directory):
    os.makedirs(directory)

#out = uf.ImportAxopatchData(file)

f = h5py.File(datafile, 'r')

i1data = f['LowPassSegmentation/i1/']
i2data = f['LowPassSegmentation/i2/']

ind1 = np.uint64(i1data['CommonIndex'][:])
ind2 = np.uint64(i2data['CommonIndex'][:])
indexes1 = np.zeros(len(i1data['DwellTime']), dtype=np.bool)
indexes1[ind1] = 1
indexes2 = np.zeros(len(i2data['DwellTime']), dtype=np.bool)
indexes2[ind2] = 1

print(len(i1data['OnlyIndex'][:]))
print(len(i2data['OnlyIndex'][:]))

#indexes1 = np.ones(len(i1data['DwellTime']), dtype=np.bool)
#indexes2 = np.ones(len(i2data['DwellTime']), dtype=np.bool)

# Scatter Plot, Dwell Time, CurrentDrop
fig = plt.figure(1)
ax1 = fig.add_subplot(111)
plt.title('Ionic Dwell Time vs. Delta I')
ax1.scatter(i1data['DwellTime'][indexes1]*1e6, i1data['DeltaI'][indexes1]*1e9, c='b')
ax1.set_xlabel('Time [us]')
ax1.set_ylabel('Ionic Current Drop [nA]', color='b')
ax2 = ax1.twinx()
ax2.scatter(i2data['DwellTime'][indexes2]*1e6, i2data['DeltaI'][indexes2]*1e9, c='r')
ax2.set_ylabel('Transverse Current Drop [nA]', color='r')
fig.savefig(directory + os.sep + 'ScatterPlot.eps')
fig.savefig(directory + os.sep + 'ScatterPlot.png', dpi=150)

# Histogram Dwell Time
bins = 100
fig2 = plt.figure(2)
ax3 = fig2.add_subplot(211)
ax4 = fig2.add_subplot(212, sharex=ax3)
ax3.hist(i1data['DwellTime'][indexes1]*1e6, bins, color='b')
ax4.hist(i2data['DwellTime'][indexes2]*1e6, bins, color='r')
ax4.set_xlabel('Dwell Time [us]')
ax4.set_ylabel('Transverse Counts')
ax3.set_ylabel('Ionic Counts')
ax3.set_title('Dwell Time Scatter')
fig2.savefig(directory + os.sep + 'DwellHist.eps')
fig2.savefig(directory + os.sep + 'DwellHist.png', dpi=150)

# Histogram Current Drop
fig3 = plt.figure(3)
ax5 = fig3.add_subplot(211)
ax6 = fig3.add_subplot(212)
ax5.hist(i1data['DeltaI'][indexes1]*1e9, bins, color='b')
ax6.hist(i2data['DeltaI'][indexes2]*1e9, bins, color='r')
ax6.set_xlabel('Current Drop [nA]')
ax6.set_ylabel('Transverse Counts')
ax5.set_ylabel('Ionic Counts')
ax5.set_title('Current Drop Scatter')
fig3.savefig(directory + os.sep + 'CurrentDropHist.eps')
fig3.savefig(directory + os.sep + 'CurrentDropHist.png', dpi=150)

plt.show()

