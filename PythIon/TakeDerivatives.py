# Script to look at derivative of one channel
import UsefulFunctions as uf
import numpy as np
import scipy
import scipy.signal as sig
import os
from scipy import io
from scipy import signal
from PyQt5 import QtGui, QtWidgets
import matplotlib.pyplot as plt
from numpy import linalg as lin
import pyqtgraph as pg
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

filetoload='/Volumes/backup/2016/Michael/Axopatch/21112016/17B_10mMCis100mMtransKCl_80mer.dat'
PartToConsider=np.array([21.542, 21.566])
out = uf.ImportAxopatchData(filetoload)

partinsamples=np.int64(np.round(out['samplerate']*PartToConsider))

i1part=out['i1'][partinsamples[0]:partinsamples[1]]
i2part=out['i2'][partinsamples[0]:partinsamples[1]]
t=self.t[partinsamples[0]:partinsamples[1]]

plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(t, i1part, 'b')
plt.title('i1 vs. i2')
plt.ylabel('Ionic Current [A]')
ax = plt.gca()
ax.set_xticklabels([])

plt.subplot(2, 1, 2)
plt.plot(t, i2part, 'r')
plt.xlabel('time (s)')
plt.ylabel('Transverse Current [A]')

plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(t, i1part, 'b')
plt.title('i1 vs. its derivative')
plt.ylabel('Ionic Current [A]')
ax = plt.gca()
ax.set_xticklabels([])

plt.subplot(2, 1, 2)
plt.plot(t[:-1], np.diff(i1part), 'y')
plt.xlabel('time (s)')
plt.ylabel('d(Ionic Current [A])/dt')

plt.figure(3)
plt.subplot(2, 1, 1)
plt.plot(t, i2part, 'r')
plt.title('i2 vs. its derivative')
plt.ylabel('Transverse Current [A]')
ax = plt.gca()
ax.set_xticklabels([])

plt.subplot(2, 1, 2)
plt.plot(t[:-1], np.diff(i2part), 'y')
plt.xlabel('time (s)')
plt.ylabel('d(Transverse Current [A])/dt')

plt.show()