import numpy as np
import pyqtgraph as pg
import UsefulFunctions as uf
from matplotlib.ticker import NullFormatter
from matplotlib import pyplot as plt
import h5py
import os
from matplotlib.backends.backend_pdf import PdfPages


expname = '10mMCis100mMTrans80mer_BigFileEverything'
buffer=500
file = '/Users/migraf/Desktop/Temp/Axo Data/17B_10mMCis100mMtransKCl_80mer_2.dat'
datafile = '/Users/migraf/Desktop/Temp/Axo Data/17B_10mMCis100mMtransKCl_80mer_2_OriginalDB.hdf5'

directory = (str(os.path.split(datafile)[0]) + os.sep + expname + '_SavedImages')

if not os.path.exists(directory):
    os.makedirs(directory)

file = '/Users/migraf/Desktop/Temp/Axo Data/17B_10mMCis100mMtransKCl_80mer_2.dat'
datafile = '/Users/migraf/Desktop/Temp/Axo Data/17B_10mMCis100mMtransKCl_80mer_2_OriginalDB.hdf5'

out = uf.ImportAxopatchData(file)

f = h5py.File(datafile, 'r')

i1data = f['LowPassSegmentation/i1/']
i2data = f['LowPassSegmentation/i2/']


#Plot All Common Events
pp=PdfPages(directory + os.sep + 'SavedEventsCommon.pdf')
ind1 = np.uint64(i1data['CommonIndex'][:])
ind2 = np.uint64(i2data['CommonIndex'][:])
t = np.arange(0, len(out['i1']))
t = t / out['samplerate'] * 1e3
for eventnumber in range(len(ind1)):
    parttoplot = np.arange(i1data['StartPoints'][eventnumber] - buffer, i1data['EndPoints'][eventnumber] + buffer, 1, dtype=np.uint64)
    parttoplot2 = np.arange(i2data['StartPoints'][eventnumber] - buffer, i2data['EndPoints'][eventnumber] + buffer, 1, dtype=np.uint64)
    fit1 = np.concatenate([np.ones(buffer)*i1data['LocalBaseline'][eventnumber],
                     np.ones(i1data['EndPoints'][eventnumber]-i1data['StartPoints'][eventnumber])*i1data['DeltaI'][eventnumber],
                     np.ones(buffer)*i1data['LocalBaseline'][eventnumber]])
    fit2 = np.concatenate([np.ones(buffer)*i2data['LocalBaseline'][eventnumber],
                     np.ones(i2data['EndPoints'][eventnumber]-i2data['StartPoints'][eventnumber])*i2data['DeltaI'][eventnumber],
                     np.ones(buffer)*i2data['LocalBaseline'][eventnumber]])
    fig = uf.PlotEvent(t[parttoplot], t[parttoplot2], out['i1'][parttoplot], out['i2'][parttoplot2], fit1, fit2)
    pp.savefig(fig)
    print('{} out of {} saved!'.format(str(eventnumber), str(len(ind1))))
    fig.clear()
    plt.close(fig)
pp.close()


#Plot All Common Events
pp=PdfPages(directory + os.sep + 'SavedEventsOnlyi1.pdf')
ind1 = np.uint64(i1data['CommonIndex'][:])
ind2 = np.uint64(i2data['CommonIndex'][:])
t = np.arange(0, len(out['i1']))
t = t / out['samplerate'] * 1e3
for eventnumber in range(len(ind1)):
    parttoplot = np.arange(i1data['StartPoints'][eventnumber] - buffer, i1data['EndPoints'][eventnumber] + buffer, 1, dtype=np.uint64)
    parttoplot2 = np.arange(i2data['StartPoints'][eventnumber] - buffer, i2data['EndPoints'][eventnumber] + buffer, 1, dtype=np.uint64)
    fit1 = np.concatenate([np.ones(buffer)*i1data['LocalBaseline'][eventnumber],
                     np.ones(i1data['EndPoints'][eventnumber]-i1data['StartPoints'][eventnumber])*i1data['DeltaI'][eventnumber],
                     np.ones(buffer)*i1data['LocalBaseline'][eventnumber]])
    fit2 = np.concatenate([np.ones(buffer)*i2data['LocalBaseline'][eventnumber],
                     np.ones(i2data['EndPoints'][eventnumber]-i2data['StartPoints'][eventnumber])*i2data['DeltaI'][eventnumber],
                     np.ones(buffer)*i2data['LocalBaseline'][eventnumber]])
    fig = uf.PlotEvent(t[parttoplot]*1e6, t[parttoplot2]*1e6, out['i1'][parttoplot]*1e9, out['i2'][parttoplot2]*1e9, fit1*1e9, fit2*1e9)
    pp.savefig(fig)
    print('{} out of {} saved!'.format(str(eventnumber), str(len(ind1))))
    fig.clear()
    plt.close(fig)
pp.close()