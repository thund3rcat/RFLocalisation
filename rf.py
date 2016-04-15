# import modules
import time as t
import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import *
        
class RFear:

    # Statisches Attribut / Klassenattribut
    
    
  
    # Konstruktor
    def __init__(self,f):                                       
        self.sdr = RtlSdr()
        self.sdr.sample_rate = 2.4e6
        self.sdr.center_freq = f
        self.freq = f
        self.sdr.gain = 1

    def __del__(self):
        del(self.sdr)
        
    def getFreq(self):
        print(str(self.freq) + " Hz")

    def setFreq(self,f):
        self.freq = f
        self.sdr.center_freq = f
    
    # get Power Spectral Density Live Plot    
    def getPSD(self):
        plt.ion()
        plt.figure()
        plt.show()
        drawing = True
        while drawing:
            try:
                # Busy-wait for keyboard interrupt (Ctrl+C)
                plt.clf()
                plt.axis([self.freq/1e6 - 1.5, self.freq/1e6 + 1.5, -50, 30])
                samples = self.sdr.read_samples(256 * 1024)
                # use matplotlib to estimate and plot the PSD
                plt.psd(samples, NFFT=1024, Fs=self.sdr.sample_rate / 1e6, Fc=self.freq / 1e6)
                plt.xlabel("Frequency (MHz)")
                plt.ylabel("Relative power (dB)")
                plt.draw()
                time.sleep(0.01)
            except KeyboardInterrupt:
                print "Liveplot stopped by User."
                drawing = False

    
    # measure power values of specified frequency for a certain time
    def getPower(self, time, size):
        elapsed_time = 0      
        powerstack = []
        timestack = []      
        while elapsed_time < time :
            pass  # Busy-wait for keyboard interrupt (Ctrl+C)  
            start_calctime = t.time()
	    # read samples    
            samples = self.sdr.read_samples(size * 1024)
            # use matplotlib to estimate the PSD and save the max power
            power, freqs = plt.psd(samples, NFFT=1024, Fs=self.sdr.sample_rate / 1e6, Fc=self.freq / 1e6)
	    powerstack.append(max(power))
            calctime = t.time() - start_calctime
            timestack.append(calctime)
            elapsed_time = elapsed_time + calctime
        
        calctime = np.mean(calctime)    
        plt.clf()
        plt.plot(powerstack,"bo")
        plt.figure()
        plt.plot(timestack,"g^")
        plt.show()
        return calctime, elapsed_time
        
