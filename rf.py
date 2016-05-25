# import modules
# import modules
import time as t
import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import *
from scipy import signal

class RFear:
    """A simple class to compute PSD with a DVBT-dongle"""
    # Statisches Attribut / Klassenattribut
    pass
    
    # Konstruktor
    def __init__(self, *args):                                       
        self.__sdr = RtlSdr()
        self.__sdr.sample_rate = 2.4e6        
        self.__sdr.gain = 1
        self.__freq = []
        self.set_freq(self, *args)
    
    # Destruktor
    def __del__(self):
        del(self.__sdr)
    
    # Methoden    
    def get_freq(self):
        return self.__freq

    def set_freq(self, *args):  
        self.__freq[:] = []
        for arg in args:
            self.__freq.append(arg)
        if not isinstance(self.__freq[0], float):
            self.__freq = self.__freq[1:]
        self.__sdr.center_freq = np.mean(self.__freq)
              
    def get_psd(self): # get Power Spectral Density Live Plot 
        plt.ion()      # turn interactive mode on
        plt.show()
        drawing = True
        while drawing:
            try:
                # Busy-wait for keyboard interrupt (Ctrl+C)
                plt.clf()
                plt.axis([self.__sdr.center_freq/1e6 - 1.5, self.__sdr.center_freq/1e6 + 1.5, -50, 30])
                samples = self.__sdr.read_samples(256 * 1024)
                # use matplotlib to estimate and plot the PSD
                plt.psd(samples, NFFT=1024, Fs=self.__sdr.sample_rate/1e6, Fc=self.__sdr.center_freq/1e6)
                plt.xlabel("Frequency (MHz)")
                plt.ylabel("Relative power (dB)")
                plt.draw()
                t.sleep(0.01)
            except KeyboardInterrupt:
                plt.draw()
                print "Liveplot interrupted by user"
                drawing = False

    def find_peaks(self, power, freqs, interval=80):
        l = len(self.__freq)
        power = power.tolist()
        freqsTemp = freqs.tolist()
        freqsTemp = [float(xx) for xx in freqs]
        freqsTemp = [round(xxx, 1) for xxx in freqs]
        # print "\n Freq Temp: \n"
        # print freqsTemp 
        # print "\n Length of Freq vector: \n"
        # print len(freqs)
        pmax = []
        for i in range(l):
            x = freqsTemp.index(self.__freq[i]/1e6)
            pmax.append(max(power[x: x + interval]))
        return pmax
    def find_peaks2(self, power, freqs, interval=20):
        return max(power)  

    # measure power values of specified frequency for a certain time
    def get_power(self, time=10, size=256):
        elapsed_time = 0      
        powerstack = []
        timestack = []      
        while elapsed_time < time: 
            start_calctime = t.time()
	    # read samples    
            samples = self.__sdr.read_samples(size * 1024)
            # use matplotlib to estimate the PSD and save the max power
            power, freqs = plt.psd(samples, NFFT=1024, Fs=self.__sdr.sample_rate/1e6, Fc=self.__sdr.center_freq/1e6)
            print "\n Freq: \n" 
            print freqs
            # print freqs
	    powerstack.append(self.find_peaks(power, freqs))
            t.sleep(0.005)
            calctime = t.time() - start_calctime
            timestack.append(calctime)
            elapsed_time = elapsed_time + calctime
        print "Finished"
        calctime = np.mean(calctime)   
        #text = "Rechenzeit pro Update: " + calctime + " sec\n Gesamtzeit: " + elapsed_time + " sec"
        print "Number of samples per update:\n"
        print size*1024
        print "\n"
        print "Calculation time per update (sec):\n"
        print calctime
        print "\n"
        print "Total time (sec):\n"
        print elapsed_time 
        plt.clf()
        plt.axis([0, len(powerstack), -50, 30])
        plt.plot(10*np.log10(powerstack), "o")
        plt.xlabel("Updates")
        plt.ylabel("Maximum power (dB)")
        plt.figure()
        plt.plot(timestack, "g^")
        plt.xlabel("Updates")
        plt.ylabel("Computation time (sec)")
        plt.show()
        return powerstack

    def rpi_get_power(self, printing=0, size=256):
        #"""Routine for Raspberry Pi: """
        running = True
        pmax = []
        while running:
            try:
                samples = self.__sdr.read_samples(size * 1024)        
                power, freqs = plt.psd(samples, NFFT=1024, Fs=self.__sdr.sample_rate/1e6, Fc=self.__sdr.center_freq/1e6)
                pmax.append(self.find_peaks(power, freqs))
                if printing:
                    print self.find_peaks(power, freqs)
                    print "\n"
                else:
                    pass
            except KeyboardInterrupt:
                print "Process interrupted by user"
                return pmax
                running = False
        

    def write_to_file(self, l, text, filename="Experiments"):
        f = open(filename, "a")
        f.write(t.ctime() + "\n")
        f.write(text + "\n")
        f.write(str(l))
        f.write("\n\n")
        f.close()
        
