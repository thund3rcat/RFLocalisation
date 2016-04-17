# import modules
import time as t
import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import *
        
class RFear:

    # Statisches Attribut / Klassenattribut
    pass
    
  
    # Konstruktor
    def __init__(self, *args):                                       
        self.__sdr = RtlSdr()
        self.__sdr.sample_rate = 2.4e6        
        self.__sdr.gain = 1
        self.__freq = []
        self.setFreq(self,*args)
    
    # Destruktor
    def __del__(self):
        del(self.__sdr)
    
    # Methoden    
    def getFreq(self):
        return self.__freq

    def setFreq(self,*args):  
        self.__freq[:] = []
        for arg in args:
            self.__freq.append(arg)
        if not isinstance(self.__freq[0],float):
            self.__freq = self.__freq[1:]
        self.__sdr.center_freq = np.mean(self.__freq)
              
    def getPSD(self): # get Power Spectral Density Live Plot 
        plt.ion()
        plt.figure()
        plt.show()
        while True:
            plt.clf()
            plt.axis([self.__sdr.center_freq/1e6 - 1.5, self.__sdr.center_freq/1e6 + 1.5, -50, 30])
            samples = self.__sdr.read_samples(256 * 1024)
            # use matplotlib to estimate and plot the PSD
            plt.psd(samples, NFFT=1024, Fs=self.__sdr.sample_rate / 1e6, Fc=self.__sdr.center_freq / 1e6)
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Relative power (dB)")
            plt.draw()
            t.sleep(0.05)
                   
    def getPower(self, time, size): # measure power values of specified frequency for a certain time
        elapsed_time = 0      
        powerstack = []
        timestack = []      
        while elapsed_time < time :  
            start_calctime = t.time()
	    # read samples    
            samples = self.__sdr.read_samples(size * 1024)
            # use matplotlib to estimate the PSD and save the max power
            power, freqs = plt.psd(samples, NFFT=1024, Fs=self.__sdr.sample_rate / 1e6, Fc=self.__sdr.center_freq / 1e6)
	    powerstack.append(max(power))
            calctime = t.time() - start_calctime
            timestack.append(calctime)
            elapsed_time = elapsed_time + calctime
        
        calctime = np.mean(calctime)    
        plt.clf()
        plt.axis([self.__sdr.center_freq/1e6 - 1.5, self.__sdr.center_freq/1e6 + 1.5, -50, 30])
        plt.plot(powerstack,"bo")
        plt.figure()
        plt.plot(timestack,"g^")
        plt.show()
        return calctime, elapsed_time
        
