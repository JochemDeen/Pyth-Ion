import numpy as np
import scipy
import scipy.signal as sig
import UsefulFunctions as uf
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('small')

Tk().withdraw()
os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "python" to true' ''')

expname = 'Gradient'

filename='/Users/migraf/SWITCHdrive/PhD/Pyth-Ion-Fork/PythIon/Sample Data/IVs/Axopatch2Channel_Ch1Sweep.dat'
output = uf.OpenFile(filename)
directory = (str(os.path.split(filename)[0]) + os.sep + expname + '_SavedImages')
AllData = uf.MakeIVData(output, delay=2)

figIV = plt.figure(2)
ax1IV = figIV.add_subplot(111)
ax1IV = uf.PlotIV(output, AllData, current='i1', unit=1e9, axis=ax1IV, WithFit=0)
figIV.tight_layout()

# Save Figures
#figIV.savefig(directory + os.sep + str(os.path.split(filename)[1]) + 'IV_i1.png', dpi=150)
#figIV.savefig(directory + os.sep + str(os.path.split(filename)[1]) + 'IV_i1.eps')

figIV.show()
plt.show()