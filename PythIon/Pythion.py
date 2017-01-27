#!/usr/bin/python
# -*- coding: utf8 -*-
import sys
import numpy as np
from scipy import ndimage
import os
from scipy import signal
from scipy import io as spio
from UserInterface import *
#from plotgui4k import *
#from plotguiretina import *
import pyqtgraph as pg
import pyqtgraph.exporters
from pyqtgraph.dockarea import *
import pandas.io.parsers
import pandas as pd
from abfheader import *
from CUSUMV2 import detect_cusum
from PoreSizer import *
from batchinfo import *
import UsefulFunctions as uf
import scipy
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import time
import h5py
from timeit import default_timer as timer
import platform


class GUIForm(QtGui.QMainWindow):


    def __init__(self, master=None):
        ####Setup GUI and draw elements from UI file#########
        QtGui.QMainWindow.__init__(self,master)
        self.ui = Ui_PythIon()
        self.ui.setupUi(self)

        ##########Linking buttons to main functions############
        self.ui.IVxaxis.currentIndexChanged.connect(self.IVAxis)
        self.ui.IVyaxis.currentIndexChanged.connect(self.IVAxis)
        self.ui.loadbutton.clicked.connect(self.getfile)
        self.ui.analyzebutton.clicked.connect(self.analyze)
        self.ui.cutbutton.clicked.connect(self.cut)
        self.ui.baselinebutton.clicked.connect(self.baselinecalc)
        self.ui.clearscatterbutton.clicked.connect(self.clearscatter)
        self.ui.invertbutton.clicked.connect(self.invertdata)
        #self.ui.invertbutton.clicked.connect(self.makeIV)
        self.ui.nextfilebutton.clicked.connect(self.nextfile)
        self.ui.previousfilebutton.clicked.connect(self.previousfile)
        self.ui.gobutton.clicked.connect(self.inspectevent)
        self.ui.previousbutton.clicked.connect(self.previousevent)
        self.ui.nextbutton.clicked.connect(self.nextevent)
        self.ui.fitbutton.clicked.connect(self.CUSUM)
        self.ui.Poresizeraction.triggered.connect(self.sizethepore)
        self.ui.ndChannel.clicked.connect(self.Plot)
        self.ui.makeIVButton.clicked.connect(self.makeIV)
        self.ui.actionSave_All.triggered.connect(self.SaveAllFigures)
        self.ui.groupBox_5.clicked.connect(self.customCond)
        self.ui.customCurrent.valueChanged.connect(self.UpdateIV)
        self.ui.customVoltage.valueChanged.connect(self.UpdateIV)
        self.ui.customConductanceSpinBox.valueChanged.connect(self.UpdateIV)
        self.ui.concentrationValue.valueChanged.connect(self.UpdateIV)
        self.ui.porelengthValue.valueChanged.connect(self.UpdateIV)
        self.ui.actionUse_Clipping.triggered.connect(self.DisplaySettings)
        self.ui.actionUse_Downsampling.triggered.connect(self.DisplaySettings)
        self.ui.actionSave_IV_Data.triggered.connect(self.SaveIVData)
        self.ui.actionPlot_Common_Events.triggered.connect(self.EventFiltering)
        self.ui.actionPlot_i2_detected_only.triggered.connect(self.EventFiltering)
        self.ui.actionPlot_i1_detected_only.triggered.connect(self.EventFiltering)

        self.ui.actionUse_Clipping.setChecked(False)
        #        self.ui.actionBatch_Process.triggered.connect(self.batchinfodialog)
        self.ui.plotBoth.clicked.connect(self.Plot)
        ###### Setting up plotting elements and their respective options######
        self.ui.signalplot.setBackground('w')
        self.ui.scatterplot.setBackground('w')
        self.ui.eventplot.setBackground('w')
        self.ui.frachistplot.setBackground('w')
        self.ui.delihistplot.setBackground('w')
        self.ui.dwellhistplot.setBackground('w')
        self.ui.dthistplot.setBackground('w')
        self.ui.voltageplotwin.setBackground('w')
        self.ui.ivplot.setBackground('w')
        self.ui.cutData.setBackground('w')
#        self.ui.PSDplot.setBackground('w')
        self.ui.AxopatchGroup.setVisible(0)

        self.ui.label_2.setText('Output Samplerate (kHz)' + str(pg.siScale(np.float(self.ui.outputsamplerateentry.text()))[1]))
        self.p1 = self.ui.signalplot
        self.transverseAxis = pg.ViewBox()
        self.transverseAxisVoltage = pg.ViewBox()
        self.transverseAxisEvent = pg.ViewBox()

        self.p1.enableAutoRange(axis='y')
        self.p1.disableAutoRange(axis='x')
        self.p1.setDownsampling(ds=False, auto=True, mode='subsample')
        self.p1.setClipToView(False)

        self.voltagepl = self.ui.voltageplotwin
        self.voltagepl.enableAutoRange(axis='y')
        self.voltagepl.disableAutoRange(axis='x')
        self.voltagepl.setDownsampling(ds=True, auto=True, mode='subsample')
        self.voltagepl.setClipToView(False)
        self.voltagepl.setXLink(self.p1)

        self.ivplota = self.ui.ivplot
        #self.ivplot.setLabel('bottom', text='Current', units='A')
        #self.ivplot.setLabel('left', text='Voltage', units='V')
        #self.ivplot.enableAutoRange(axis = 'x')
        self.psdplot = self.ui.powerSpecPlot
        self.psdplot.setBackground('w')
        self.cutplot = self.ui.cutData
        #self.cutplot.setLabel('bottom', text='Time', units='s')
        #self.cutplot.setLabel('left', text='Voltage', units='V')
        #self.cutplot.enableAutoRange(axis = 'x')

        self.w1 = self.ui.scatterplot.addPlot()
        self.p2 = pg.ScatterPlotItem()
        self.p2.sigClicked.connect(self.clicked)
        self.w1.addItem(self.p2)
        self.w1.setLabel('bottom', text='Time', units=u'μs')
        self.w1.setLabel('left', text='Fractional Current Blockage')
        self.w1.setLogMode(x=True,y=False)
        self.w1.showGrid(x=True, y=True)
        self.cb = pg.ColorButton(self.ui.scatterplot, color=(0,0,255,50))
        self.cb.setFixedHeight(30)
        self.cb.setFixedWidth(30)
        self.cb.move(0,250)
        self.cb.show()

        self.w2 = self.ui.frachistplot.addPlot()
        self.w2.setLabel('bottom', text='Fractional Current Blockage')
        self.w2.setLabel('left', text='Counts')

        self.w3 = self.ui.delihistplot.addPlot()
        self.w3.setLabel('bottom', text='ΔI', units ='A')
        self.w3.setLabel('left', text='Counts')

        self.w4 = self.ui.dwellhistplot.addPlot()
        self.w4.setLabel('bottom', text='Log Dwell Time', units = 'μs')
        self.w4.setLabel('left', text='Counts')

        self.w5 = self.ui.dthistplot.addPlot()
        self.w5.setLabel('bottom', text='dt', units = 's')
        self.w5.setLabel('left', text='Counts')

#        self.w6 = self.ui.PSDplot.addPlot()
#        self.w6.setLogMode(x = True, y = True)
#        self.w6.setLabel('bottom', text='Frequency (Hz)')
#        self.w6.setLabel('left', text='PSD (pA^2/Hz)')

        self.p3 = self.ui.eventplot
        self.p3.hideAxis('bottom')
        self.p3.hideAxis('left')

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.logo = ndimage.imread(dir_path + os.sep + "pythionlogo.png")
        self.logo = np.rot90(self.logo,-1)
        self.logo = pg.ImageItem(self.logo)
        self.p3.addItem(self.logo)
        self.p3.setAspectLocked(True)

        self.ui.conductanceText.setText('Conductance: ')
        self.ui.resistanceText.setText('Resistance: ')
        self.ui.poresizeOutput.setText('Pore Size: ')
        self.useCustomConductance = 0
        self.conductance = 1e-9
        self.ui.porelengthValue.setOpts(value=0.7E-9, suffix='m', siPrefix=True, dec=True, step=1e-9, minStep=1e-9)
        self.ui.concentrationValue.setOpts(value=10, suffix='S/m', siPrefix=True, dec=True, step=10e-3, minStep=10e-3)
        self.ui.customCurrent.setOpts(value=10e-9, suffix='A', siPrefix=True, dec=True, step=10e-3, minStep=10e-3)
        self.ui.customVoltage.setOpts(value=500e-3, suffix='V', siPrefix=True, dec=True, step=10e-3, minStep=10e-3)
        self.ui.customConductanceSpinBox.setOpts(value=10e-9/500e-3, suffix='S', siPrefix=True, dec=True, step=10e-3, minStep=10e-3)

        self.ui.LP_a.setOpts(value=0.999, suffix='', siPrefix=False, dec=True, step=1e-3, minStep=1e-4)
        self.ui.LP_S.setOpts(value=5, suffix='x STD', siPrefix=True, dec=True, step=10e-3, minStep=10e-3)
        self.ui.LP_E.setOpts(value=0, suffix='x STD', siPrefix=True, dec=True, step=10e-3, minStep=10e-3)
        self.ui.LP_eventlengthThresh.setOpts(value=1e-3, suffix='s', siPrefix=True, dec=True, step=10e-3, minStep=10e-3)

        self.ui.LP_a_2.setOpts(value=0.999, suffix='', siPrefix=False, dec=True, step=1e-3, minStep=1e-4)
        self.ui.LP_S_2.setOpts(value=5, suffix='x STD', siPrefix=True, dec=True, step=10e-3, minStep=10e-3)
        self.ui.LP_E_2.setOpts(value=0, suffix='x STD', siPrefix=True, dec=True, step=10e-3, minStep=10e-3)
        self.ui.LP_eventlengthThresh_2.setOpts(value=1e-3, suffix='s', siPrefix=True, dec=True, step=10e-3, minStep=10e-3)

        ####### Initializing various variables used for analysis##############
        self.NumberOfEvents=0
        self.AnalysisResults = {}
        self.sig = 'i1'
        self.xaxisIV=self.ui.IVxaxis.currentIndex()
        self.yaxisIV=self.ui.IVyaxis.currentIndex()
        self.Channel2=0
        self.direc=[]
        self.lr=[]
        self.lastevent=[]
        self.lastClicked=[]
        self.hasbaselinebeenset=0
        self.lastevent=0
        self.deli=[]
        self.frac=[]
        self.dwell=[]
        self.dt=[]
        self.catdata=[]
        self.colors=[]
        self.sdf = pd.DataFrame(columns = ['fn','color','deli','frac',
            'dwell','dt','startpoints','endpoints'])

    def Load(self, loadandplot = True):
        print('File Adress: {}'.format(self.datafilename))
        print('Timestamp: {}'.format(uf.creation_date(self.datafilename)))

        self.count=0
        if hasattr(self, 'pp'):
            if hasattr(self.pp,'close'):
                self.pp.close()
        self.catdata=[]
        self.batchinfo = pd.DataFrame(columns = list(['cutstart', 'cutend']))
        self.p3.clear()
        self.p3.setLabel('bottom', text='Current', units='A', unitprefix = 'n')
        self.p3.setLabel('left', text='', units = 'Counts')
        self.p3.setAspectLocked(False)
        self.p1.enableAutoRange(axis='x')

        colors = np.array(self.sdf.color)
        for i in range(len(colors)):
            colors[i] = pg.Color(colors[i])

        self.p2.setBrush(colors, mask=None)
        self.ui.eventinfolabel.clear()
        self.totalplotpoints=len(self.p2.data)
        self.ui.eventnumberentry.setText(str(0))
        self.hasbaselinebeenset=0
        self.threshold=np.float64(self.ui.thresholdentry.text())*10**-9
        self.ui.filelabel.setText(self.datafilename)
        self.LPfiltercutoff = np.float64(self.ui.LPentry.text())*1000
        self.outputsamplerate = np.float64(self.ui.outputsamplerateentry.text())*1000 #use integer multiples of 4166.67 ie 2083.33 or 1041.67
        print()
        if str(os.path.splitext(self.datafilename)[1])=='.dat':
            print('Loading Axopatch Data')
            self.out=uf.ImportAxopatchData(self.datafilename)
            self.matfilename = str(os.path.splitext(self.datafilename)[0])
            self.outputsamplerate=self.out['samplerate']
            self.ui.outputsamplerateentry.setText(str(self.out['samplerate']))
            if self.out['graphene']:
                self.ui.AxopatchGroup.setVisible(1)
            else:
                self.ui.AxopatchGroup.setVisible(0)

        if str(os.path.splitext(self.datafilename)[1]) == '.log':
            print('Loading Chimera File')
            self.out = uf.ImportChimeraData(self.datafilename)
            self.matfilename = str(os.path.splitext(self.datafilename)[0])
            if self.out['type'] == 'ChimeraNotRaw':
                self.data = self.out['current']
                print(str(self.data.shape))
                self.vdata = self.out['voltage']
            else:
                Wn = round(self.LPfiltercutoff/(self.out['samplerate']/2), 4)
                b,a = signal.bessel(4, Wn, btype='low');
                self.out['lowpassedData'] = signal.filtfilt(b,a,self.out['current'])
                self.data = self.out['lowpassedData']
                self.vdata = np.ones(len(self.data)) * self.out['voltage']

        if str(os.path.splitext(self.datafilename)[1])=='.opt':
            self.data = np.fromfile(self.datafilename, dtype = np.dtype('>d'))
            self.matfilename = str(os.path.splitext(self.datafilename)[0])  
            
            try:
                self.mat = spio.loadmat(self.matfilename + '_inf')  
                samplerate = np.float64(self.mat['samplerate'])
                lowpass = np.float64(self.mat['filterfreq'])
                print(samplerate)
                print(lowpass)
            except TypeError:
                pass
            
            if self.outputsamplerate > 250e3:
                    print('sample rate can not be >250kHz for axopatch files, displaying with a rate of 250kHz')
                    self.outputsamplerate  = 250e3
#            self.data=self.data*10**9

            if self.LPfiltercutoff < 100e3:
                Wn = round(self.LPfiltercutoff/(100*10**3/2),4)
                b,a = signal.bessel(4, Wn, btype='low');
                self.data = signal.filtfilt(b,a,self.data)
            else:
                print('Filter value too high, data not filtered')

        if str(os.path.splitext(self.datafilename)[1])=='.txt':
            self.data=pandas.io.parsers.read_csv(self.datafilename,skiprows=1)
#            self.data=np.reshape(np.array(self.data),np.size(self.data))*10**9
            self.data=np.reshape(np.array(self.data),np.size(self.data))
            self.matfilename=str(os.path.splitext(self.datafilename)[0])

        if str(os.path.splitext(self.datafilename)[1])=='.npy':
            self.data = np.load(self.datafilename)
            self.matfilename=str(os.path.splitext(self.datafilename)[0])

        if str(os.path.splitext(self.datafilename)[1])=='.abf':
            f = open(self.datafilename, "rb")  # reopen the file
            f.seek(6144, os.SEEK_SET)
            self.data = np.fromfile(f, dtype = np.dtype('<i2'))
            self.matfilename=str(os.path.splitext(self.datafilename)[0])
            self.header = read_header(self.datafilename)
            self.samplerate = 1e6/self.header['protocol']['fADCSequenceInterval']
            self.telegraphmode = int(self.header['listADCInfo'][0]['nTelegraphEnable'])
            if self.telegraphmode == 1:
                self.abflowpass = self.header['listADCInfo'][0]['fTelegraphFilter']
                self.gain = self.header['listADCInfo'][0]['fTelegraphAdditGain']
            else:
                self.gain = 1
                self.abflowpass = self.samplerate
                
            self.data=self.data.astype(float)*(20./(65536*self.gain))*10**-9                
 
            if len(self.header['listADCInfo']) == 2:
                self.v = self.data[1::2]*self.gain/10
                self.data = self. data[::2]
            else:
                self.v = [] 
               
                
            if self.outputsamplerate > self.samplerate:
                    print('output samplerate can not be higher than samplerate, resetting to original rate')
                    self.outputsamplerate  = self.samplerate
                    self.ui.outputsamplerateentry.setText(str((round(self.samplerate)/1000)))
            if self.LPfiltercutoff >= self.abflowpass:
                    print('Already LP filtered lower than or at entry, data will not be filtered')
                    self.LPfiltercutoff  = self.abflowpass
                    self.ui.LPentry.setText(str((round(self.LPfiltercutoff)/1000)))
            else:
                Wn = round(self.LPfiltercutoff/(100*10**3/2),4)
                b,a = signal.bessel(4, Wn, btype='low');
                self.data = signal.filtfilt(b,a,self.data)

                
            tags = self.header['listTag']
            for tag in tags:
                if tag['sComment'][0:21] == "Holding on 'Cmd 0' =>":
                    cmdv = tag['sComment'][22:]
#                    cmdv = [int(s) for s in cmdv.split() if s.isdigit()]
                    cmdt = tag ['lTagTime']/self.outputsamplerate
                    self.p1.addItem(pg.InfiniteLine(cmdt))
#                    cmdtext = pg.TextItem(text = str(cmdv)+' mV')
                    cmdtext = pg.TextItem(text = str(cmdv))
                    self.p1.addItem(cmdtext)
                    cmdtext.setPos(cmdt,np.max(self.data))

        self.t=np.arange(0, len(self.out['i1']))
        self.t=self.t/self.out['samplerate']

        self.ui.label_2.setText('Output Samplerate ' + str(pg.siScale(np.float(self.outputsamplerate))[1]))

        if loadandplot == True:
            self.Plot()

    def Plot(self):
        # if self.hasbaselinebeenset==0:
        #     self.baseline=np.median(self.out['i1'])
        #     self.var=np.std(self.out['i1'])
        # self.ui.eventcounterlabel.setText('Baseline='+str(round(self.baseline*10**9, 2))+' nA')

        if not self.ui.plotBoth.isChecked():
            if self.ui.ndChannel.isChecked():
                self.sig = 'i2'
                self.sig2 = 'i1'
            else:
                self.sig = 'i1'
                self.sig2 = 'i2'
        print(self.out[self.sig].shape)
        if not self.ui.actionDon_t_Plot_if_slow.isChecked():
            if self.ui.plotBoth.isChecked():
                uf.DoublePlot(self)
            else:
                uf.PlotSingle(self)

    def getfile(self):
        datafilenametemp = QtGui.QFileDialog.getOpenFileName(parent=self, caption='Open file', directory=str(self.direc), filter="Amplifier Files(*.log *.opt *.npy *.txt *.abf *.dat)")
        if not datafilenametemp[0]=='':
            self.datafilename=datafilenametemp[0]
            self.direc=os.path.dirname(self.datafilename)
            self.Load()


    def SaveIVData(self):
        uf.ExportIVData(self)

    def analyze(self):
        if 1:
            self.coefficients={}
            self.coefficients['i1'] = {'a': np.float(self.ui.LP_a.value()), 'E': np.float(self.ui.LP_E.value()),
                                 'S': np.float(self.ui.LP_S.value()),
                                 'eventlengthLimit': np.float(self.ui.LP_eventlengthThresh.value()) * self.out[
                                     'samplerate']}
            self.coefficients['i2'] = {'a': np.float(self.ui.LP_a_2.value()), 'E': np.float(self.ui.LP_E_2.value()),
                                 'S': np.float(self.ui.LP_S_2.value()),
                                 'eventlengthLimit': np.float(self.ui.LP_eventlengthThresh_2.value()) * self.out[
                                     'samplerate']}
            chan = ['i1', 'i2']
            start1 = timer()
            for sig in chan:
                self.AnalysisResults[sig] = {}
                self.AnalysisResults[sig]['RoughEventLocations'] = uf.RecursiveLowPassFast(self.out[sig], self.coefficients[sig])
                if 0:
                    self.AnalysisResultsUp[sig] = {}
                    self.AnalysisResultsUp[sig]['RoughEventLocations'] = uf.RecursiveLowPassFastUp(self.out[self.sig], self.coefficients[sig])


            end1 = timer()
            print('The Low-pass took {} s on both channels.'.format(str(start1-end1)))
            self.sig = 'i1'
            uf.AddInfoAfterRecursive(self)
            self.sig = 'i2'
            uf.AddInfoAfterRecursive(self)
            end2 = timer()
            print('Adding Info took {} s on both channels.'.format(str(end2-end1)))

            #uf.SavingAndPlottingAfterRecursive(self)
            uf.SaveToHDF5(self)
            end3 = timer()
            print('Saving took {} s on both channels.'.format(str(end3-end2)))

            if 'i1' in self.AnalysisResults and 'i2' in self.AnalysisResults:
                (self.CommonIndexes, self.OnlyIndexes) = uf.CombineTheTwoChannels(self.matfilename + '_OriginalDB.hdf5')
                print('Two channels are combined')
                uf.EditInfoText(self)
                self.EventFiltering(self)
            end4 = timer()
            print('Combining took {} s on both channels.'.format(str(end4-end3)))
        else:
            return

    def inspectevent(self, clicked = []):
        if self.ui.actionPlot_Common_Events.isChecked():
            uf.PlotEventDoubleFit(self, clicked)
        elif self.ui.actionPlot_i1_detected_only.isChecked() or self.ui.actionPlot_i2_detected_only.isChecked():
            uf.PlotEventDouble(self)

    def nextevent(self):
        eventnumber=np.int(self.ui.eventnumberentry.text())

        if eventnumber>=self.NumberOfEvents:
            eventnumber=0
        else:
            eventnumber=np.int(self.ui.eventnumberentry.text())+1
        self.ui.eventnumberentry.setText(str(eventnumber))
        self.inspectevent()

    def previousevent(self):
        eventnumber=np.int(self.ui.eventnumberentry.text())-1
        self.ui.eventnumberentry.setText(str(eventnumber))
        self.inspectevent()

    def cut(self):
        
        ###### first check to see if cutting############

        if self.lr==[]:
            ######## if no cutting window exists, make one##########
            self.lr = pg.LinearRegionItem()
            self.lr.hide()

            ##### detect clears and auto-position window around the clear#####
            clears = np.where(np.abs(self.data) > self.baseline + 10*self.var)[0]
            if clears != []:
                clearstarts = clears[0]
                try:
                    clearends = clearstarts + np.where((self.data[clearstarts:-1] > self.baseline) &
                    (self.data[clearstarts:-1] < self.baseline+self.var))[0][10000]
                except:
                    clearends = -1
                clearstarts = np.where(self.data[0:clearstarts] > self.baseline)
                try:
                    clearstarts = clearstarts[0][-1]
                except:
                    clearstarts = 0

                self.lr.setRegion((self.t[clearstarts],self.t[clearends]))

            self.p1.addItem(self.lr)
            self.lr.show()


        #### if cut region has been set, cut region and replot remaining data####
        else:
            cutregion = self.lr.getRegion()
            self.p1.clear()
            self.data = np.delete(self.data,np.arange(np.int(cutregion[0]*self.outputsamplerate),np.int(cutregion[1]*self.outputsamplerate)))

            self.t=np.arange(0,len(self.data))
            self.t=self.t/self.outputsamplerate

            if self.hasbaselinebeenset==0:
                self.baseline = np.median(self.data)
                self.var=np.std(self.data)
                self.ui.eventcounterlabel.setText('Baseline='+str(round(self.baseline*10**9,2))+' nA')

            self.p1.plot(self.t,self.data,pen='b')
            if str(os.path.splitext(self.datafilename)[1]) != '.abf':
                self.p1.addLine(y=self.baseline,pen='g')
                self.p1.addLine(y=self.threshold,pen='r')
            self.lr=[]
#            self.p1.autoRange()
            self.p3.clear()
            aphy, aphx = np.histogram(self.data, bins = len(self.data)/1000,range = [np.min(self.data),np.max(self.data)])
            aphhist = pg.BarGraphItem(height = aphy, x0 = aphx[:-1], x1 = aphx[1:],brush = 'b', pen = None)
            self.p3.addItem(aphhist)
            self.p3.setXRange(np.min(self.data), np.max(self.data))
            
            cf = pd.DataFrame([cutregion], columns = list(['cutstart', 'cutend']))
            self.batchinfo = self.batchinfo.append(cf, ignore_index = True)

    def baselinecalc(self):
        if self.lr==[]:
            self.p1.clear()
            self.lr = pg.LinearRegionItem()
            self.lr.hide()
            self.p1.addItem(self.lr)

#            self.p1.plot(self.t[::100],self.data[::100],pen='b')
            self.p1.plot(self.t,self.data,pen='b')
            self.lr.show()

        else:
            calcregion=self.lr.getRegion()
            self.p1.clear()

            self.baseline=np.median(self.data[np.arange(np.int(calcregion[0]*self.outputsamplerate),np.int(calcregion[1]*self.outputsamplerate))])
            self.var=np.std(self.data[np.arange(np.int(calcregion[0]*self.outputsamplerate),np.int(calcregion[1]*self.outputsamplerate))])
#            self.p1.plot(self.t[::10][2:][:-2],self.data[::10][2:][:-2],pen='b')
            self.p1.plot(self.t,self.data,pen='b')
            self.p1.addLine(y=self.baseline,pen='g')
            self.p1.addLine(y=self.threshold,pen='r')
            self.lr=[]
            self.hasbaselinebeenset=1
            self.ui.eventcounterlabel.setText('Baseline='+str(round(self.baseline*10**9,2))+' nA')
            self.p1.autoRange()

    def clearscatter(self):
        self.p2.setData(x=[],y=[])
        self.lastevent=[]
        self.ui.scatterplot.update()
        self.w2.clear()
        self.w3.clear()
        self.w4.clear()
        self.w5.clear()
        self.sdf = pd.DataFrame(columns = ['fn','color','deli','frac',
            'dwell','dt','startpoints','endpoints'])

    def deleteevent(self):
        eventnumber = np.int(self.ui.eventnumberentry.text())
        firstindex = self.sdf.fn[self.sdf.fn == self.matfilename].index[0]
        if eventnumber > self.numberofevents:
            eventnumber = self.numberofevents-1
            self.ui.eventnumberentry.setText(str(eventnumber))
        self.deli=np.delete(self.deli,eventnumber)
        self.dwell=np.delete(self.dwell,eventnumber)
        self.dt=np.delete(self.dt,eventnumber)
        self.frac=np.delete(self.frac,eventnumber)
        self.startpoints=np.delete(self.startpoints, eventnumber)
        self.endpoints=np.delete(self.endpoints, eventnumber)
        self.p2.data=np.delete(self.p2.data,firstindex + eventnumber)

        self.numberofevents = len(self.dt)
        self.ui.eventcounterlabel.setText('Events:'+str(self.numberofevents))

        self.sdf = self.sdf.drop(firstindex + eventnumber).reset_index(drop = True)
        self.inspectevent()

        self.w2.clear()
        self.w3.clear()
        self.w4.clear()
        self.w5.clear()
        colors = self.sdf.color
        for i, x in enumerate(colors):
            fracy, fracx = np.histogram(self.sdf.frac[self.sdf.color == x], bins=np.linspace(0, 1, int(self.ui.fracbins.text())))
            deliy, delix = np.histogram(self.sdf.deli[self.sdf.color == x], bins=np.linspace(float(self.ui.delirange0.text())*10**-9, float(self.ui.delirange1.text())*10**-9, int(self.ui.delibins.text())))
            dwelly, dwellx = np.histogram(np.log10(self.sdf.dwell[self.sdf.color == x]), bins=np.linspace(float(self.ui.dwellrange0.text()), float(self.ui.dwellrange1.text()), int(self.ui.dwellbins.text())))
            dty, dtx = np.histogram(self.sdf.dt[self.sdf.color == x], bins=np.linspace(float(self.ui.dtrange0.text()), float(self.ui.dtrange1.text()), int(self.ui.dtbins.text())))

#            hist = pg.PlotCurveItem(fracy, fracx , stepMode = True, fillLevel=0, brush = x, pen = 'k')
#            self.w2.addItem(hist)

            hist = pg.BarGraphItem(height = fracy, x0 = fracx[:-1], x1 = fracx[1:], brush = x)
            self.w2.addItem(hist)

#            hist = pg.PlotCurveItem(delix, deliy , stepMode = True, fillLevel=0, brush = x, pen = 'k')
#            self.w3.addItem(hist)

            hist = pg.BarGraphItem(height = deliy, x0 = delix[:-1], x1 = delix[1:], brush = x)
            self.w3.addItem(hist)
#            self.w3.autoRange()
            self.w3.setRange(xRange = [float(self.ui.delirange0.text())*10**-9, float(self.ui.delirange1.text())*10**-9])

#            hist = pg.PlotCurveItem(dwellx, dwelly , stepMode = True, fillLevel=0, brush = x, pen = 'k')
#            self.w4.addItem(hist)

            hist = pg.BarGraphItem(height = dwelly, x0 = dwellx[:-1], x1 = dwellx[1:], brush = x)
            self.w4.addItem(hist)

#            hist = pg.PlotCurveItem(dtx, dty , stepMode = True, fillLevel=0, brush = x, pen = 'k')
#            self.w5.addItem(hist)

            hist = pg.BarGraphItem(height = dty, x0 = dtx[:-1], x1 = dtx[1:], brush = x)
            self.w5.addItem(hist)

        self.save()
        uf.SaveToHDF5(self)

    def invertdata(self):
        self.p1.clear()
        self.data=-self.data

        if self.hasbaselinebeenset==0:
            self.baseline=np.median(self.data)
            self.var=np.std(self.data)

#        self.p1.plot(self.t[::10],self.data[::10],pen='b')
        self.p1.plot(self.t,self.data,pen='b')
        self.p1.addLine(y=self.baseline,pen='g')
        self.p1.addLine(y=self.threshold,pen='r')
        self.p1.autoRange()

    def clicked(self, plot, points):
        for i, p in enumerate(self.p2.points()):
            if p.pos() == points[0].pos():
                clickedindex = i

        if self.sdf.fn[clickedindex] != self.matfilename:
            print('Event is from an earlier file, not clickable')

        else:
            self.inspectevent(clickedindex)

    def concatenatetext(self):
        if self.direc==[]:
            textfilenames = QtGui.QFileDialog.getOpenFileNames(self, 'Open file','*.txt')
            self.direc=os.path.dirname(str(textfilenames[0]))
        else:
            textfilenames =QtGui.QFileDialog.getOpenFileNames(self, 'Open file',self.direc,'*.txt')
            self.direc=os.path.dirname(str(textfilenames[0]))
        i=0
        while i<len(textfilenames):
            temptextdata=np.fromfile(str(textfilenames[i]),sep='\t')
            temptextdata=np.reshape(temptextdata,(len(temptextdata)/4,4))
            if i==0:
                newtextdata=temptextdata
            else:
                newtextdata=np.concatenate((newtextdata,temptextdata))
            i=i+1

        newfilename = QtGui.QFileDialog.getSaveFileName(self, 'New File name',self.direc,'*.txt')
        np.savetxt(str(newfilename),newtextdata,delimiter='\t')

    def nextfile(self):
        if str(os.path.splitext(self.datafilename)[1])=='.log':
            startindex=self.matfilename[-6::]
            filebase=self.matfilename[0:len(self.matfilename)-6]
            nextindex=str(int(startindex)+1)
            while os.path.isfile(filebase+nextindex+'.log')==False:
                nextindex=str(int(nextindex)+1)
                if int(nextindex)>int(startindex)+1000:
                    print('no such file')
                    break
            if os.path.isfile(filebase+nextindex+'.log')==True:
                self.datafilename=(filebase+nextindex+'.log')
                self.Load()

        if str(os.path.splitext(self.datafilename)[1])=='.abf':
            startindex=self.matfilename[-4::]
            filebase=self.matfilename[0:len(self.matfilename)-4]
            nextindex=str(int(startindex)+1).zfill(4)
            while os.path.isfile(filebase+nextindex+'.abf')==False:
                nextindex=str(int(nextindex)+1).zfill(4)
                if int(nextindex)>int(startindex)+1000:
                    print('no such file')
                    break
            if os.path.isfile(filebase+nextindex+'.abf')==True:
                self.datafilename=(filebase+nextindex+'.abf')
                self.Load()

    def previousfile(self):
        if str(os.path.splitext(self.datafilename)[1])=='.log':
            startindex=self.matfilename[-6::]
            filebase=self.matfilename[0:len(self.matfilename)-6]
            nextindex=str(int(startindex)-1)
            while os.path.isfile(filebase+nextindex+'.log')==False:
                nextindex=str(int(nextindex)-1)
                if int(nextindex)<int(startindex)-1000:
                    print('no such file')
                    break
            if os.path.isfile(filebase+nextindex+'.log')==True:
                self.datafilename=(filebase+nextindex+'.log')
                self.Load()

        if str(os.path.splitext(self.datafilename)[1])=='.abf':
            startindex=self.matfilename[-4::]
            filebase=self.matfilename[0:len(self.matfilename)-4]
            nextindex=str(int(startindex)-1).zfill(4)
            while os.path.isfile(filebase+nextindex+'.abf')==False:
                nextindex=str(int(nextindex)-1).zfill(4)
                if int(nextindex)<int(startindex)-1000:
                    print('no such file')
                    break
            if os.path.isfile(filebase+nextindex+'.abf')==True:
                self.datafilename=(filebase+nextindex+'.abf')
                self.Load()

    def savetrace(self):
        self.data.astype('d').tofile(self.matfilename+'_trace.bin')

    def showcattrace(self):
        eventbuffer=np.int(self.ui.eventbufferentry.value())
        numberofevents=len(self.dt)

        self.p1.clear()
        eventtime = [0]
        for i in range(numberofevents):
            if i<numberofevents-1:
                if endpoints[i]+eventbuffer>startpoints[i+1]:
                    print('overlapping event')
                else:
                    eventdata = self.data[startpoints[i]-eventbuffer:endpoints[i]+eventbuffer]
                    fitdata = np.concatenate((np.repeat(np.array([self.localBaseline[i]]),eventbuffer),np.repeat(np.array([self.localBaseline[i]-self.deli[i]]),endpoints[i]-startpoints[i]),np.repeat(np.array([self.localBaseline[i]]),eventbuffer)),0)
                    eventtime = np.arange(0,len(eventdata)) + .75*eventbuffer + eventtime[-1]
                    self.p1.plot(eventtime/self.outputsamplerate, eventdata,pen='b')
                    self.p1.plot(eventtime/self.outputsamplerate, fitdata,pen=pg.mkPen(color=(173,27,183),width=2))

        self.p1.autoRange()

    def savecattrace(self):
        eventbuffer=np.int(self.ui.eventbufferentry.value())
        numberofevents=len(self.dt)
        self.catdata=self.data[startpoints[0]-eventbuffer:endpoints[0]+eventbuffer]
        self.catfits=np.concatenate((np.repeat(np.array([self.baseline]),eventbuffer),np.repeat(np.array([
            self.baseline-self.deli[0]]),endpoints[0]-startpoints[0]),
            np.repeat(np.array([self.baseline]),eventbuffer)),0)

        for i in range(numberofevents):
            if i<numberofevents-1:
                if endpoints[i]+eventbuffer>startpoints[i+1]:
                    print('overlapping event')
                else:
                    self.catdata=np.concatenate((self.catdata,self.data[startpoints[i]-eventbuffer:endpoints[i]+eventbuffer]),0)
                    self.catfits=np.concatenate((self.catfits,np.concatenate((np.repeat(np.array([self.baseline]),eventbuffer),np.repeat(np.array([
                        self.baseline-self.deli[i]]),endpoints[i]-startpoints[i]),np.repeat(np.array([self.baseline]),eventbuffer)),0)),0)

        self.tcat=np.arange(0,len(self.catdata))
        self.tcat=self.tcat/self.outputsamplerate
        self.catdata=self.catdata[::10]
        self.catdata.astype('d').tofile(self.matfilename+'_cattrace.bin')

    def keyPressEvent(self, event):
        key = event.key()
        if key == QtCore.Qt.Key_Up:
            self.nextfile()
        if key == QtCore.Qt.Key_Down:
            self.previousfile()
        if key == QtCore.Qt.Key_Right:
            self.nextevent()
        if key == QtCore.Qt.Key_Left:
            self.previousevent()
    #    if key == QtCore.Qt.Key_Return:
    #        self.Load()
        if key == QtCore.Qt.Key_Space:
            self.analyze()
        if key == QtCore.Qt.Key_Delete:
            self.deleteevent()
        if key == QtCore.Qt.Key_S:
            self.skeypressed()

    def saveeventfits(self):
        eventbuffer=np.int(self.ui.eventbufferentry.value())
        numberofevents=len(self.dt)
        self.catdata=self.data[startpoints[0]-eventbuffer:endpoints[0]+eventbuffer]
        self.catfits=np.concatenate((np.repeat(np.array([self.baseline]),eventbuffer),np.repeat(np.array([
            self.baseline-self.deli[0]]),endpoints[0]-startpoints[0]),
            np.repeat(np.array([self.baseline]),eventbuffer)),0)

        for i in range(numberofevents):
            if i<numberofevents-1:
                if endpoints[i]+eventbuffer>startpoints[i+1]:
                    print('overlapping event')
                else:
                    self.catdata=np.concatenate((self.catdata,self.data[startpoints[i]-eventbuffer:endpoints[i]+eventbuffer]),0)
                    self.catfits=np.concatenate((self.catfits,np.concatenate((np.repeat(np.array([self.baseline]),eventbuffer),np.repeat(np.array([
                        self.baseline-self.deli[i]]),endpoints[i]-startpoints[i]),np.repeat(np.array([self.baseline]),eventbuffer)),0)),0)

        self.tcat=np.arange(0,len(self.catdata))
        self.tcat=self.tcat/self.outputsamplerate
        self.catfits.astype('d').tofile(self.matfilename+'_cattrace.bin')

    def CUSUM(self):
        self.p1.clear()
        self.p1.setDownsampling(ds = False)
        cusum = detect_cusum(self.data, basesd = self.var, dt = 1/self.outputsamplerate, 
                             threshhold  = np.float64(self.ui.thresholdentry.text()),
                             stepsize = np.float64(self.ui.levelthresholdentry.text()), 
                             minlength = 10)
        np.savetxt(self.matfilename+'_Levels.txt', np.abs(cusum['jumps']*10**12),delimiter='\t')

        self.p1.plot(self.t[2:][:-2],self.data[2:][:-2],pen='b')

        self.w3.clear()
        amp = np.abs(cusum['jumps']*10**12)
        ampy, ampx = np.histogram(amp, bins=np.linspace(float(self.ui.delirange0.text()), float(self.ui.delirange1.text()), int(self.ui.delibins.text())))
        hist = pg.BarGraphItem(height = ampy, x0 = ampx[:-1], x1 = ampx[1:], brush = 'b')
        self.w3.addItem(hist)
#        self.w3.autoRange()
        self.w3.setRange(xRange = [np.min(ampx),np.max(ampx)])

        cusumlines = np.array([]).reshape(0,2)
        for i,level in enumerate(cusum['CurrentLevels']):
            y = 2*[level]
            x = cusum['EventDelay'][i:i+2]
            self.p1.plot(y = y, x = x, pen = 'r')
            cusumlines = np.concatenate((cusumlines,np.array(zip(x,y))))
            try:
                y = cusum['CurrentLevels'][i:i+2]
                x = 2*[cusum['EventDelay'][i+1]]
                self.p1.plot(y = y, x = x, pen = 'r')
                cusumlines = np.concatenate((cusumlines,np.array(zip(x,y))))
            except Exception:
                pass
            
        cusumlines.astype('d').tofile(self.matfilename+'_cusum.bin')
        self.savetrace()

    def savetarget(self):
        self.batchinfo = self.batchinfo.append(pd.DataFrame({'deli':self.deli,
                    'frac':self.frac,'dwell':self.dwell,'dt':self.dt, 
                    'startpoints':startpoints,'endpoints':endpoints}), ignore_index=True)
        self.batchinfo.to_pickle(self.matfilename+'batchinfo.pkl')

    def batchinfodialog(self):
        self.bp = batchprocesser()
        self.bp.show()
        
        QtCore.QObject.connect(self.bp.uibp.okbutton, QtCore.SIGNAL('clicked()'), self.batchprocess)
        
    def batchprocess(self):
        global endpoints, startpoints
        
        self.p1.setDownsampling(ds = False)
        self.mindwell = np.float64(self.bp.uibp.mindwellbox.text())
        self.minfrac = np.float64(self.bp.uibp.minfracbox.text())
        self.minlevelt = np.float64(self.bp.uibp.minleveltbox.text())*10**-6
        self.samplerate = self.bp.uibp.sampratebox.text()
        self.LPfiltercutoff = self.bp.uibp.LPfilterbox.text()
        self.ui.outputsamplerateentry.setText(self.samplerate)
        self.ui.LPentry.setText(self.LPfiltercutoff)
        cusumstep = np.float64(self.bp.uibp.cusumstepentry.text())
        cusumthresh = np.float64(self.bp.uibp.cusumthreshentry.text())
        self.bp.destroy()   
        self.p1.clear()
        
        try:
            ######## attempt to open dialog from most recent directory########
            self.filelist = QtGui.QFileDialog.getOpenFileNames(self,'Select Files',self.direc,("*.pkl"))
            self.direc=os.path.dirname(self.filelist[0])
        except TypeError:
            ####### if no recent directory exists open from working directory##
            self.direc==[]
            self.filelist = QtGui.QFileDialog.getOpenFileNames(self, 'Select Files',os.getcwd(),("*.pkl"))
            self.direc=os.path.dirname(self.filelist[0])
        except IOError:
            #### if user cancels during file selection, exit loop#############
            return

        eventbuffer=np.int(self.ui.eventbufferentry.vlaue())
        eventtime = [0]
        ll = np.array([])


        for f in self.filelist: 
            batchinfo = pd.read_pickle(f)
            try:
                self.datafilename = f[:-13] + '.opt'
                self.Load(loadandplot = False)
            except IOError:
                self.datafilename = f[:-13] + '.log'
                self.Load(loadandplot = False)
                
            
            try:
                cs = batchinfo.cutstart[np.isfinite(batchinfo.cutstart)]
                ce = batchinfo.cutend[np.isfinite(batchinfo.cutend)]
                for i, cut in enumerate(cs):
                    self.data = np.delete(self.data,np.arange(np.int(cut*self.outputsamplerate),np.int(ce[i]*self.outputsamplerate)))
            except TypeError:
                pass
             
             
            self.deli = np.array(batchinfo.deli[np.isfinite(batchinfo.deli)])
            self.frac = np.array(batchinfo.frac[np.isfinite(batchinfo.frac)])
            self.dwell = np.array(batchinfo.dwell[np.isfinite(batchinfo.dwell)])
            self.dt = np.array(batchinfo.dt[np.isfinite(batchinfo.dt)])
            startpoints = np.array(batchinfo.startpoints[np.isfinite(batchinfo.startpoints)])
            endpoints = np.array(batchinfo.endpoints[np.isfinite(batchinfo.endpoints)])
            
            for i,dwell in enumerate(self.dwell):
                print(str(i) + '/' + str(len(self.dwell)))
                toffset = (eventtime[-1] + .75*eventbuffer)/self.outputsamplerate
                if i < len(self.dt)-1 and dwell > self.mindwell and self.frac[i] >self.minfrac:
                    if endpoints[i]+eventbuffer>startpoints[i+1]:
                        print('overlapping event')
                    else:
                        eventdata = self.data[startpoints[i]-eventbuffer:endpoints[i]+eventbuffer]
                        eventtime = np.arange(0,len(eventdata)) + .75*eventbuffer + eventtime[-1]
                        self.p1.plot(eventtime/self.outputsamplerate, eventdata,pen='b')
                        cusum = detect_cusum(eventdata, basesd = np.std(eventdata[0:eventbuffer])
                            , dt = 1/self.outputsamplerate, threshhold  = cusumthresh
                            , stepsize = cusumstep, minlength = self.minlevelt*self.outputsamplerate, maxstates = 10)
                        
                        while len(cusum['CurrentLevels']) < 3:
                            cusumthresh = cusumthresh *.9
                            cusumstep = cusumstep * .9
                            cusum = detect_cusum(eventdata, basesd = np.std(eventdata[0:eventbuffer])
                                , dt = 1/self.outputsamplerate, threshhold  = cusumthresh
                                , stepsize = cusumstep, minlength = self.minlevelt*self.outputsamplerate, maxstates = 10)
                      
#                        print len(cusum['CurrentLevels'])

                        
#                        if np.max(cusum['CurrentLevels'])-np.min(cusum['CurrentLevels']) == 0:
#                            cusum = detect_cusum(eventdata, basesd = np.std(eventdata)
#                                , dt = 1/self.outputsamplerate, threshhold  = cusumthresh/10
#                                , stepsize = cusumstep/10, minlength = self.minlevelt*self.outputsamplerate)
                            
                        ll = np.concatenate((ll,[(np.max(cusum['CurrentLevels'])-np.min(cusum['CurrentLevels']))/np.max(cusum['CurrentLevels'])]))
                        cusumthresh = cusum['Threshold']
                        cusumstep = cusum['stepsize']
                                                    
                        
                        for j,level in enumerate(cusum['CurrentLevels']):
                            self.p1.plot(y = 2*[level], x = toffset + cusum['EventDelay'][j:j+2], pen = pg.mkPen( 'r', width = 5))
                            try:
                                self.p1.plot(y = cusum['CurrentLevels'][j:j+2], x = toffset + 2*[cusum['EventDelay'][j+1]], pen = pg.mkPen( 'r', width = 5))
                            except Exception:
                                pass

        np.savetxt(self.matfilename+'llDB.txt',ll,delimiter='\t')
        self.p1.autoRange()
        
        print('\007')
        
    def IVAxis(self):
        self.xaxisIV = self.ui.IVxaxis.currentIndex()
        self.yaxisIV = self.ui.IVyaxis.currentIndex()

    def sizethepore(self):
        self.ps = PoreSizer()
        self.ps.show()

    def makeIV(self):
        if self.yaxisIV == 0:
            xlab = 'i1'
        if self.yaxisIV == 1:
            xlab = 'i2'
        if self.xaxisIV == 0:
            ylab = 'v1'
        if self.xaxisIV == 1:
            ylab = 'v2'
        self.cutplot.clear()
        (AllData, a) = uf.CutDataIntoVoltageSegments(self.out, delay=1, plotSegments=1, x=xlab,
                                                                                y=ylab, extractedSegments = self.cutplot)
        if AllData is not 0:
            # Make IV
            (self.IVData, b) = uf.MakeIV(AllData, plot=0)
            # Fit IV
            self.ivplota.clear()
            (self.FitValues, iv) = uf.FitIV(self.IVData, x=xlab, y=ylab, iv=self.ivplota)
            self.conductance = self.FitValues['Slope']
            self.UpdateIV()
        # Update Conductance

    def UpdateIV(self):
        self.ui.conductanceText.setText('Conductance: ' + pg.siFormat(self.conductance, precision=5, suffix='S', space=True, error=None, minVal=1e-25, allowUnicode=True))
        self.ui.resistanceText.setText('Resistance: ' + pg.siFormat(1/self.conductance, precision=5, suffix='Ohm', space=True, error=None, minVal=1e-25, allowUnicode=True))
        if self.useCustomConductance:
            self.ui.customConductanceSpinBox.setValue(self.ui.customCurrent.value()/self.ui.customVoltage.value())
            valuetoupdate=np.float(self.ui.customConductanceSpinBox.value())
        else:
            valuetoupdate=self.conductance
        print(self.ui.porelengthValue.value())
        print(valuetoupdate)
        print(self.ui.concentrationValue.value())
        size=uf.CalculatePoreSize(valuetoupdate, self.ui.porelengthValue.value(), self.ui.concentrationValue.value())
        self.ui.poresizeOutput.setText('Pore Size: ' + pg.siFormat(size, precision=5, suffix='m', space=True, error=None, minVal=1e-25, allowUnicode=True))

    def customCond(self):
        if self.ui.groupBox_5.isChecked():
            self.useCustomConductance = 1
            self.ui.conductanceText.setEnabled(False)
            self.ui.resistanceText.setEnabled(False)
            self.UpdateIV()
        else:
            self.useCustomConductance = 0
            self.ui.conductanceText.setEnabled(True)
            self.ui.resistanceText.setEnabled(True)
            self.UpdateIV()

    def SaveAllFigures(self):
        self.pp.close()

    def DisplaySettings(self):
        if self.ui.actionUse_Clipping.isChecked():
            self.p1.setClipToView(True)
            self.Plot()
        else:
            self.p1.setClipToView(False)
            self.Plot()
        if self.ui.actionUse_Downsampling.isChecked():
            self.p1.setDownsampling(ds=True, auto=True, mode='subsample')
            self.Plot()
        else:
            self.p1.setDownsampling(ds=False, auto=True, mode='subsample')
            self.Plot()

    def skeypressed(self):
        if self.ui.ivplot.underMouse():
            uf.MatplotLibIV(self)
        if self.ui.signalplot.underMouse():
            if not self.count:
                # PDF to save images:
                filename = os.path.splitext(os.path.basename(self.datafilename))[0]
                dirname = os.path.dirname(self.datafilename)
                self.count = 1
                while os.path.isfile(dirname + os.sep + filename + '_AllSavedImages_' + str(self.count) + '.pdf'):
                    self.count += 1
                self.pp = PdfPages(dirname + os.sep + filename + '_AllSavedImages_' + str(self.count) + '.pdf')
            print('Added to PDF_' + str(self.count))
            uf.MatplotLibCurrentSignal(self)
            self.ui.signalplot.setBackground('g')
            time.sleep(1)
            self.ui.signalplot.setBackground('w')
        if self.ui.eventplot.underMouse():
            print('Event Plot Saved...')
            uf.SaveEventPlotMatplot(self)

    def EventFiltering(self, who):
        if self.ui.actionPlot_Common_Events.isChecked():
            self.NumberOfEvents = len(self.CommonIndexes['i1'])
        if self.ui.actionPlot_i1_detected_only.isChecked():
            self.NumberOfEvents = len(self.OnlyIndexes['i1'])
        if self.ui.actionPlot_i2_detected_only.isChecked():
            self.NumberOfEvents = len(self.OnlyIndexes['i2'])

    def Test(self):
        print('Yeeeeaahhh')

def QCloseEvent(w):
    print('Application closed, config saved...')

def start():
    app = QtGui.QApplication(sys.argv)
    myapp = GUIForm()
    myapp.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    global myapp
    app = QtGui.QApplication(sys.argv)
    myapp = GUIForm()
    myapp.show()
    sys.exit(app.exec_())

