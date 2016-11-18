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


def Reshape1DTo2D(inputarray, buffersize):
    npieces = np.uint16(len(inputarray)/buffersize)
    voltages = np.array([], dtype=np.float64)
    currents = np.array([], dtype=np.float64)
    #print(npieces)

    for i in range(1, npieces+1):
        if i % 2 == 1:
            currents = np.append(currents, inputarray[(i-1)*buffersize:i*buffersize-1], axis=0)
            #print('Length Currents: {}'.format(len(currents)))
        else:
            voltages = np.append(voltages, inputarray[(i-1)*buffersize:i*buffersize-1], axis=0)
            #print('Length Voltages: {}'.format(len(voltages)))

    v1 = np.ones((1, len(voltages)), dtype=np.float64)
    i1 = np.ones((1, len(currents)), dtype=np.float64)
    v1[:]=voltages
    i1[:]=currents

    out = {'v1': v1, 'i1': i1}
    print('Currents:' + str(v1.shape))
    print('Voltages:' + str(i1.shape))
    return out

def CalculatePoreSize(G, L, s):
    return (G+np.sqrt(G*(G+16*L*s/np.pi)))/(2*s)

def ImportAxopatchData(datafilename):
    x=np.fromfile(datafilename, np.dtype('>f4'))
    f=open(datafilename, 'rb')
    graphene=0
    for i in range(0, 8):
        a=str(f.readline())
        if 'Acquisition' in a or 'Sample Rate' in a:
            samplerate=int(''.join(i for i in a if i.isdigit()))/1000
        if 'I_Graphene' in a:
            graphene=1
            print('This File Has a Graphene Channel!')
    end = len(x)
    if graphene:
        #pore current
        i1 = x[250:end-3:4]
        #graphene current
        i2 = x[251:end-2:4]
        #pore voltage
        v1 = x[252:end-1:4]
        #graphene voltage
        v2 = x[253:end:4]
        output={'type': 'Axopatch', 'graphene': 1, 'samplerate': samplerate, 'i1': i1, 'v1': v1, 'i2': i2, 'v2': v2, 'filename': datafilename}
    else:
        i1 = x[250:end-1:2]
        v1 = x[251:end:2]
        output={'type': 'Axopatch', 'graphene': 0, 'samplerate': samplerate, 'i1': i1, 'v1': v1, 'filename': datafilename}
    return output

def ImportChimeraRaw(datafilename):
    matfile=io.loadmat(str(os.path.splitext(datafilename)[0]))
    #buffersize=matfile['DisplayBuffer']
    data = np.fromfile(datafilename, np.dtype('<u2'))
    samplerate = np.float64(matfile['ADCSAMPLERATE'])
    TIAgain = np.int32(matfile['SETUP_TIAgain'])
    preADCgain = np.float64(matfile['SETUP_preADCgain'])
    currentoffset = np.float64(matfile['SETUP_pAoffset'])
    ADCvref = np.float64(matfile['SETUP_ADCVREF'])
    ADCbits = np.int32(matfile['SETUP_ADCBITS'])

    closedloop_gain = TIAgain * preADCgain
    bitmask = (2 ** 16 - 1) - (2 ** (16 - ADCbits) - 1)
    data = -ADCvref + (2 * ADCvref) * (data & bitmask) / 2 ** 16
    data = (data / closedloop_gain + currentoffset)
    data.shape = [data.shape[1], ]
    output = {'matfilename': str(os.path.splitext(datafilename)[0]),'current': data, 'voltage': np.float64(matfile['SETUP_mVoffset']), 'samplerate': samplerate, 'type': 'ChimeraRaw', 'filename': datafilename}
    return output

def ImportChimeraData(datafilename):
    matfile=io.loadmat(str(os.path.splitext(datafilename)[0]))
    samplerate=matfile['ADCSAMPLERATE']
    if samplerate<4e6:
        data=np.fromfile(datafilename, np.dtype('float64'))
        buffersize=matfile['DisplayBuffer']
        out=Reshape1DTo2D(data, buffersize)
        output={'current': out['i1'], 'voltage': out['v1'], 'samplerate':samplerate, 'type': 'ChimeraNotRaw', 'filename': datafilename}
    else:
        output = ImportChimeraRaw(datafilename)
    return output

def OpenFile(filename=''):
    if filename == '':
        datafilename = QtGui.QFileDialog.getOpenFileName()
        datafilename=datafilename[0]
        print(datafilename)
    else:
        datafilename=filename
    if datafilename[-3::] == 'dat':
        isdat = 1
        output = ImportAxopatchData(datafilename)
    else:
        isdat = 0
        output = ImportChimeraData(datafilename)
    return output

def zoom_factory(ax,base_scale = 2.):
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
            print(event.button)
        # set new limits
        ax.set_xlim([xdata - cur_xrange*scale_factor,
                     xdata + cur_xrange*scale_factor])
        ax.set_ylim([ydata - cur_yrange*scale_factor,
                     ydata + cur_yrange*scale_factor])
        plt.draw() # force re-draw

    fig = ax.get_figure() # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect('scroll_event',zoom_fun)

    #return the function
    return zoom_fun

def PlotData(output):
    if output['type'] == 'Axopatch':
        time=np.float32(np.arange(0, len(output['i1']))/output['samplerate'])
        #plot channel 1
        ch1_current = pg.PlotWidget(title="Current vs time Channel 1")
        ch1_current.plot(time, output['i1'])
        ch1_current.setLabel('left', text='Current', units='A')
        ch1_current.setLabel('bottom', text='Time', units='s')

        ch1_voltage = pg.PlotWidget(title="Voltage vs time Channel 1")
        ch1_voltage.plot(time, output['v1'])
        ch1_voltage.setLabel('left', text='Voltage', units='V')
        ch1_voltage.setLabel('bottom', text='Time', units='s')
        #ch1_voltage.setYLink(ch1_current)
        ch1_voltage.setXLink(ch1_current)
        if output['graphene']:
            # plot channel 1
            ch2_current = pg.PlotWidget(title="Current vs time Channel 2")
            ch2_current.plot(time, output['i2'])
            ch2_current.setLabel('left', text='Current', units='A')
            ch2_current.setLabel('bottom', text='Time', units='s')

            ch2_voltage = pg.PlotWidget(title="Voltage vs time Channel 2")
            ch2_voltage.plot(time, output['v2'])
            ch2_voltage.setLabel('left', text='Voltage', units='V')
            ch2_voltage.setLabel('bottom', text='Time', units='s')
            #ch2_voltage.setYLink(ch2_current)
            ch2_voltage.setXLink(ch2_current)

            fig_handles={'Ch1_Voltage': ch1_voltage, 'Ch2_Voltage': ch2_voltage, 'Ch2_Current': ch2_current, 'Ch1_Current': ch1_current}
            return fig_handles
        else:
            fig_handles = {'Ch1_Voltage': ch1_voltage, 'Ch1_Current': ch1_current, 'Ch2_Voltage': 0, 'Ch2_Current': 0}
            return fig_handles

    if output['type'] == 'ChimeraRaw':
        time=np.float32(np.arange(0, len(output['current']))/output['samplerate'])
        figure=plt.figure('Chimera Raw Current @ {} mV'.format(output['voltage']*1e3))
        plt.plot(time, output['current']*1e9)
        plt.ylabel('Current [nA]')
        plt.xlabel('Time [s]')
        figure.show()
        fig_handles = {'Fig1': figure, 'Fig2': 0, 'Zoom1': 0, 'Zoom2': 0}
        return fig_handles

    if output['type'] == 'ChimeraNotRaw':
        time=np.float32(np.arange(0, len(output['current']))/output['samplerate'])
        figure2 = plt.figure('Chimera Not Raw (Display Save Mode)')
        ax3 = plt.subplot(211)
        ax3.plot(time, output['current'] * 1e9)
        plt.ylabel('Current [nA]')
        ax4 = plt.subplot(212, sharex=ax3)
        ax4.plot(time, output['voltage'] * 1e3)
        plt.xlabel('Time [s]')
        plt.ylabel('Voltage [mV]')
        f2 = zoom_factory(ax3, 1.5)
        figure2.show()
        fig_handles = {'Fig1': 0, 'Fig2': figure2, 'Zoom1': 0, 'Zoom2': f2}
        return fig_handles

def CutDataIntoVoltageSegments(output, delay=0.7, plotSegments = 1, x='i1', y='v1', extractedSegments = ''):
    if output['type'] == 'ChimeraNotRaw':
        current = output['current']
        voltage = output['voltage']
        samplerate = output['samplerate']
    elif output['type'] == 'Axopatch' and x == 'i1' and y == 'v1':
        current = output['i1']
        voltage = output['v1']
        print('i1,v1')
        samplerate = output['samplerate']
    elif output['type'] == 'Axopatch' and output['graphene'] and x == 'i2' and y == 'v2':
        current = output['i2']
        voltage = output['v2']
        samplerate = output['samplerate']
        print('i2,v2')
    elif output['type'] == 'Axopatch' and output['graphene'] and x == 'i2' and y == 'v1':
        current = output['i2']
        voltage = output['v1']
        samplerate = output['samplerate']
        print('i2,v1')
    elif output['type'] == 'Axopatch' and output['graphene'] and x == 'i1' and y == 'v2':
        current = output['i1']
        voltage = output['v2']
        samplerate = output['samplerate']
        print('i1,v2')
    else:
        print('File doesn''t contain any IV data on the selected channel...')
        return (0, 0)

    time=np.float32(np.arange(0, len(current))/samplerate)
    delayinpoints = delay * samplerate
    diffVoltages = np.diff(voltage)
    VoltageChangeIndexes = diffVoltages
    ChangePoints = np.where(diffVoltages)[0]
    Values = voltage[ChangePoints]
    Values = np.append(Values, voltage[::-1][0])
    print('Cutting into Segments\n{} change points detected...'.format(len(ChangePoints)))
    if len(ChangePoints) is 0:
        print('Can\'t segment the file. It doesn\'t contain any voltage switches')
        return (0,0)

    #   Store All Data
    AllDataList = []
    # First
    Item={}
    Item['Voltage'] = Values[0]
    Item['CurrentTrace'] = current[0:ChangePoints[0]]
    AllDataList.append(Item)
    for i in range(1, len(Values) - 1):
        Item={}
        Item['CurrentTrace'] = current[ChangePoints[i - 1] + delayinpoints:ChangePoints[i]]
        Item['Voltage']=Values[i]
        AllDataList.append(Item)
    # Last
    Item = {}
    Item['CurrentTrace'] = current[ChangePoints[len(ChangePoints) - 1] + delayinpoints:len(current) - 1]
    Item['Voltage']= Values[len(Values) - 1]
    AllDataList.append(Item)
    if plotSegments:

        #extractedSegments = pg.PlotWidget(title="Extracted Parts")
        extractedSegments.plot(time, current, pen='b')
        extractedSegments.setLabel('left', text='Current', units='A')
        extractedSegments.setLabel('bottom', text='Time', units='s')
        # First
        extractedSegments.plot(np.arange(0,ChangePoints[0])/samplerate, current[0:ChangePoints[0]], pen='r')
        #Loop
        for i in range(1, len(Values) - 1):
            extractedSegments.plot(np.arange(ChangePoints[i - 1] + delayinpoints, ChangePoints[i]) / samplerate, current[ChangePoints[i - 1] + delayinpoints:ChangePoints[i]], pen='r')
        #Last
            extractedSegments.plot(np.arange(ChangePoints[len(ChangePoints) - 1] + delayinpoints, len(current) - 1 )/samplerate, current[ChangePoints[len(ChangePoints) - 1] + delayinpoints:len(current) - 1], pen='r')
    else:
        extractedSegments=0
    return (AllDataList, extractedSegments)

def MakeIV(CutData, plot=0):
    l=len(CutData)
    IVData={}
    IVData['Voltage'] = np.zeros(l)
    IVData['Mean'] = np.zeros(l)
    IVData['STD'] = np.zeros(l)
    count=0
    for i in CutData:
        #print('Voltage: ' + str(i['Voltage']) + ', length: ' + str(len(i['CurrentTrace'])))
        IVData['Voltage'][count] = np.float32(i['Voltage'])
        IVData['Mean'][count] = np.mean(i['CurrentTrace'])
        IVData['STD'][count] = np.std(i['CurrentTrace'])
        count+=1
    if plot:
        spacing=np.sort(IVData['Voltage'])
        iv = pg.PlotWidget(title='Current-Voltage Plot')
        err = pg.ErrorBarItem(x=IVData['Voltage'], y=IVData['Mean'], top=IVData['STD'], bottom=IVData['STD'], beam=((spacing[1]-spacing[0]))/2)
        iv.addItem(err)
        iv.plot(IVData['Voltage'], IVData['Mean'], symbol='o', pen=None)
        iv.setLabel('left', text='Current', units='A')
        iv.setLabel('bottom', text='Voltage', units='V')
    else:
        iv=0
    return (IVData, iv)

def FitIV(IVData, plot=1, x='i1', y='v1', iv=0):
    sigma_v=1e-12*np.ones(len(IVData['Voltage']))
    (a, b, sigma_a, sigma_b, b_save) = YorkFit(IVData['Voltage'], IVData['Mean'], sigma_v, IVData['STD'])
    x_fit=np.linspace(min(IVData['Voltage']), max(IVData['Voltage']), 1000)
    y_fit=scipy.polyval([b,a], x_fit)
    if plot:
        spacing=np.sort(IVData['Voltage'])
        #iv = pg.PlotWidget(title='Current-Voltage Plot', background=None)
        err = pg.ErrorBarItem(x=IVData['Voltage'], y=IVData['Mean'], top=IVData['STD'],
                              bottom=IVData['STD'], pen='b', beam=((spacing[1]-spacing[0]))/2)
        iv.addItem(err)
        iv.plot(IVData['Voltage'], IVData['Mean'], symbol='o', pen=None)
        iv.setLabel('left', text=x + ', Current', units='A')
        iv.setLabel('bottom', text=y + ', Voltage', units='V')
        iv.plot(x_fit, y_fit, pen='r')
        textval=pg.siFormat(1/b, precision=5, suffix='Ohm', space=True, error=None, minVal=1e-25, allowUnicode=True)
        textit=pg.TextItem(text=textval, color=(0, 0, 0))
        textit.setPos(min(IVData['Voltage']),max(IVData['Mean']))
        iv.addItem(textit)

    else:
        iv=0
    YorkFitValues={'x_fit': x_fit, 'y_fit': y_fit, 'Yintercept':a, 'Slope':b, 'Sigma_Yintercept':sigma_a, 'Sigma_Slope':sigma_b, 'Parameter':b_save}
    return (YorkFitValues, iv)

def MakePSD(input, samplerate, fig):
    f, Pxx_den = scipy.signal.periodogram(input, samplerate)
    #f, Pxx_den = scipy.signal.welch(input, samplerate, nperseg=10*256, scaling='spectrum')
    fig.setLabel('left', 'PSD', units='A^2/Hz')
    fig.setLabel('bottom', 'Frequency', units='Hz')
    fig.setLogMode(x=True, y=True)
    fig.plot(f, Pxx_den, pen='k')
    return (f,Pxx_den)

def YorkFit(X, Y, sigma_X, sigma_Y, r=0):
    N_itermax=10 #maximum number of interations
    tol=1e-15 #relative tolerance to stop at
    N = len(X)
    temp = np.matrix([X, np.ones(N)])
    #make initial guess at b using linear squares

    tmp = np.matrix(Y)*lin.pinv(temp)
    b_lse = np.array(tmp)[0][0]
    #a_lse=tmp(2);
    b = b_lse #initial guess
    omega_X=1/np.power(sigma_X,2)
    omega_Y=1/np.power(sigma_Y,2)
    alpha=np.sqrt(omega_X*omega_Y)
    b_save = np.zeros(N_itermax+1) #vector to save b iterations in
    b_save[0]=b

    for i in np.arange(N_itermax):
        W=omega_X*omega_Y/(omega_X+b*b*omega_Y-2*b*r*alpha)

        X_bar=np.sum(W*X)/np.sum(W)
        Y_bar=np.sum(W*Y)/np.sum(W)

        U=X-X_bar
        V=Y-Y_bar

        beta=W*(U/omega_Y+b*V/omega_X-(b*U+V)*r/alpha)

        b=sum(W*beta*V)/sum(W*beta*U)
        b_save[i+1]=b
        if np.abs((b_save[i+1]-b_save[i])/b_save[i+1]) < tol:
            break

    a=Y_bar-b*X_bar
    x=X_bar+beta
    y=Y_bar+b*beta
    x_bar=sum(W*x)/sum(W)
    y_bar=sum(W*y)/sum(W)
    u=x-x_bar
    #%v=y-y_bar
    sigma_b=np.sqrt(1/sum(W*u*u))
    sigma_a=np.sqrt(1./sum(W)+x_bar*x_bar*sigma_b*sigma_b)
    return (a, b, sigma_a, sigma_b, b_save)

def SaveFigureList(folder, list):
    filename=os.path.splitext(os.path.basename(folder))[0]
    dirname=os.path.dirname(folder)
    for i in list:
        if list[i]:
            list[i].savefig(dirname+os.sep+filename+'_'+i+'.png', format='png')
    return 0