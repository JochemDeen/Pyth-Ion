import UsefulFunctions as uf
import pyqtgraph as pg
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import scipy
import numpy as np
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.use('GTKAgg')

file='/Users/migraf/Desktop/Temp/17B_10mMCis100mMtransKCl_80mer_5.dat'
out=uf.ImportAxopatchData(file)

coefficients = {'a': 0.999, 'E': 0, 'S': 5, 'eventlengthLimit': 10e-3 * out['samplerate']}

start1 = timer()
RoughEventLocations1 = uf.RecursiveLowPass(out['i1'], coefficients)
RoughEventLocations2 = uf.RecursiveLowPass(out['i2'], coefficients)
end1 = timer()
print('Conventional Filter took :{} s'.format(str(end1 - start1)))

start2 = timer()
RoughEventLocations1 = uf.RecursiveLowPassFastUp(out['i1'], coefficients)
RoughEventLocations2 = uf.RecursiveLowPassFastUp(out['i2'], coefficients)
end2 = timer()
print('New Filter took :{} s'.format(str(end2 - start2)))



startp1=np.uint64(RoughEventLocations1[:, 0])
endp1=np.uint64(RoughEventLocations1[:, 1])
startp2=np.uint64(RoughEventLocations2[:, 0])
endp2=np.uint64(RoughEventLocations2[:, 1])


t=np.arange(0, len(out['i2']))/out['samplerate']*1e3

buffer=250

pp=PdfPages('SavedAnalysis.pdf')

for eventnumber in range(len(startp1)):
    parttoplot = np.arange(startp1[eventnumber] - buffer, endp1[eventnumber] + buffer, 1, dtype=np.uint64)
    fig = plt.figure(1, figsize=(20,7))
    plt.subplot(2, 1, 1)
    plt.cla
    plt.plot(t[parttoplot], out['i1'][parttoplot]*1e9, 'b')
    plt.ylabel('Ionic Current [nA]')
    ax = plt.gca()
    ax.set_xticklabels([])
    plt.subplot(2, 1, 2)
    plt.cla
    plt.plot(t[parttoplot], out['i2'][parttoplot]*1e9, 'r')
    plt.ylabel('Transverse Current [nA]')
    plt.xlabel('time (ms)')
    pp.savefig()
    fig.clear()
    plt.close()
pp.close()