import numpy as np
import pyqtgraph as pg
import UsefulFunctions as uf
from matplotlib.ticker import NullFormatter
from matplotlib import pyplot as plt
import h5py
import os
from matplotlib.backends.backend_pdf import PdfPages

common=1
onlyi1=1
onlyi2=1

expname = '10mMCis100mMTrans80mer_BigFileCommon'
buffer = 250
file = '/Users/migraf/Desktop/Temp/Axo Data/17B_10mMCis100mMtransKCl_80mer_2.dat'
datafile = '/Users/migraf/Desktop/Temp/Axo Data/17B_10mMCis100mMtransKCl_80mer_2_OriginalDB.hdf5'

directory = (str(os.path.split(datafile)[0]) + os.sep + expname + '_SavedImages')

if not os.path.exists(directory):
    os.makedirs(directory)

out = uf.ImportAxopatchData(file)

f = h5py.File(datafile, 'r')

i1data = f['LowPassSegmentation/i1/']
i2data = f['LowPassSegmentation/i2/']


if common:
    #Plot All Common Events
    pp=PdfPages(directory + os.sep + 'SavedEventsCommon.pdf')
    ind1 = np.uint64(i1data['CommonIndex'][:])
    ind2 = np.uint64(i2data['CommonIndex'][:])

    t = np.arange(0, len(out['i1']))
    t = t / out['samplerate'] * 1e3

    for eventnumber in range(len(ind1)):
        parttoplot = np.arange(i1data['StartPoints'][ind1[eventnumber]] - buffer, i1data['EndPoints'][ind1[eventnumber]] + buffer, 1, dtype=np.uint64)
        parttoplot2 = np.arange(i2data['StartPoints'][ind2[eventnumber]] - buffer, i2data['EndPoints'][ind2[eventnumber]] + buffer, 1, dtype=np.uint64)

        fit1 = np.concatenate([np.ones(buffer)*i1data['LocalBaseline'][ind1[eventnumber]],
                         np.ones(i1data['EndPoints'][ind1[eventnumber]]-i1data['StartPoints'][ind1[eventnumber]])*(i1data['LocalBaseline'][ind1[eventnumber]]-i1data['DeltaI'][ind1[eventnumber]]),
                         np.ones(buffer)*i1data['LocalBaseline'][ind1[eventnumber]]])

        fit2 = np.concatenate([np.ones(buffer)*i2data['LocalBaseline'][ind2[eventnumber]],
                         np.ones(i2data['EndPoints'][ind2[eventnumber]]-i2data['StartPoints'][ind2[eventnumber]])*(i2data['LocalBaseline'][ind1[eventnumber]]-i2data['DeltaI'][ind2[eventnumber]]),
                         np.ones(buffer)*i2data['LocalBaseline'][ind2[eventnumber]]])
        fig = uf.PlotEvent(t[parttoplot], t[parttoplot2], out['i1'][parttoplot], out['i2'][parttoplot2], fit1=fit1, fit2=fit2)
        pp.savefig(fig)
        print('{} out of {} saved!'.format(str(eventnumber), str(len(ind1))))
        print('Length i1: {}, Fit i1: {}'.format(len(out['i1'][parttoplot]), len(fit1)))
        print('Length i2: {}, Fit i2: {}'.format(len(out['i2'][parttoplot2]), len(fit2)))

        fig.clear()
        plt.close(fig)
    pp.close()

if onlyi1:
    #Plot All i1
    pp = PdfPages(directory + os.sep + 'SavedEventsOnlyi1.pdf')
    ind1 = np.uint64(i1data['OnlyIndex'][:])

    t = np.arange(0, len(out['i1']))
    t = t / out['samplerate'] * 1e3

    for eventnumber in range(len(ind1)):
        parttoplot = np.arange(i1data['StartPoints'][ind1[eventnumber]] - buffer, i1data['EndPoints'][ind1[eventnumber]] + buffer, 1, dtype=np.uint64)

        fit1 = np.concatenate([np.ones(buffer)*i1data['LocalBaseline'][ind1[eventnumber]],
                         np.ones(i1data['EndPoints'][ind1[eventnumber]]-i1data['StartPoints'][ind1[eventnumber]])*(i1data['LocalBaseline'][ind1[eventnumber]]-i1data['DeltaI'][ind1[eventnumber]]),
                         np.ones(buffer)*i1data['LocalBaseline'][ind1[eventnumber]]])

        fig = uf.PlotEvent(t[parttoplot], t[parttoplot], out['i1'][parttoplot], out['i2'][parttoplot], fit1=fit1)
        pp.savefig(fig)
        print('{} out of {} saved!'.format(str(eventnumber), str(len(ind1))))
        fig.clear()
        plt.close(fig)
    pp.close()

if onlyi2:
    #Plot All i2
    pp=PdfPages(directory + os.sep + 'SavedEventsOnlyi2.pdf')
    ind1 = np.uint64(i2data['OnlyIndex'][:])

    t = np.arange(0, len(out['i2']))
    t = t / out['samplerate'] * 1e3

    for eventnumber in range(len(ind1)):
        parttoplot = np.arange(i2data['StartPoints'][ind1[eventnumber]] - buffer, i2data['EndPoints'][ind1[eventnumber]] + buffer, 1, dtype=np.uint64)

        fit1 = np.concatenate([np.ones(buffer)*i2data['LocalBaseline'][ind1[eventnumber]],
                         np.ones(i2data['EndPoints'][ind1[eventnumber]]-i2data['StartPoints'][ind1[eventnumber]])*(i2data['LocalBaseline'][ind1[eventnumber]]-i2data['DeltaI'][ind1[eventnumber]]),
                         np.ones(buffer)*i2data['LocalBaseline'][ind1[eventnumber]]])

        fig = uf.PlotEvent(t[parttoplot], t[parttoplot], out['i1'][parttoplot], out['i2'][parttoplot], fit2=fit1)
        pp.savefig(fig)
        print('{} out of {} saved!'.format(str(eventnumber), str(len(ind1))))
        fig.clear()
        plt.close(fig)
    pp.close()