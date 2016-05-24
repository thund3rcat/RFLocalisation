"""This module hacks a DVBT-dongle and abuses it
as a spectrum analyzer between 23 an 1,700 MHz.
"""

# import modules
import time as t
import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import *
from scipy import signal

class RFear(object):
    """A simple class to compute PSD with a DVBT-dongle."""
    # Konstruktor
    def __init__(self, sample_rate=2.4e6, gain=1, *args):
        """Keyword-arguments:
        gain -- run rtl_test in the console to see available gains
        *args -- list of frequencies (must be in a range of __sdr.sample_rate)
        """
        self.__sdr = RtlSdr()
        self.__sdr.sample_rate = sample_rate
        self.__sdr.gain = gain
        self.__freq = []
        self.set_freq(self, *args)

    # Destruktor
    def __del__(self):
        del self.__sdr

    def get_freq(self):
        """Returns list of frequencies assigned to the object."""
        return self.__freq

    def set_freq(self, *args):
        """Defines frequencies where to listen.
        Must be in a range of __sdr.sample_rate bw.
        """
        self.__freq[:] = []
        for arg in args:
            self.__freq.append(arg)
        if not isinstance(self.__freq[0], float):
            self.__freq = self.__freq[1:]
        self.__sdr.center_freq = np.mean(self.__freq)

    def get_psd(self):
        """Get Power Spectral Density Live Plot."""
        plt.ion()      # turn interactive mode on
        plt.show()
        drawing = True
        while drawing:
            try:
                # Busy-wait for keyboard interrupt (Ctrl+C)
                plt.clf()
                plt.axis([self.__sdr.center_freq/1e6 - 1.5,
                    self.__sdr.center_freq/1e6 + 1.5, -50, 30])
                samples = self.__sdr.read_samples(256 * 1024)
                # use matplotlib to estimate and plot the PSD
                plt.psd(samples, NFFT=1024, Fs=self.__sdr.sample_rate/1e6,
                    Fc=self.__sdr.center_freq/1e6)
                plt.xlabel("Frequency (MHz)")
                plt.ylabel("Relative power (dB)")
                plt.draw()
                t.sleep(0.01)
            except KeyboardInterrupt:
                plt.draw()
                print "Liveplot interrupted by user"
                drawing = False

    def get_power(self, time=10, size=256):
        """Measure power values of specified frequency
        for a certain time and performance.
        keyword arguments:
        time -- time of measurement in seconds (default  10)
        size -- measure for length of fft (default 256)
        """
        elapsed_time = 0
        powerstack = []
        timestack = []
        while elapsed_time < time:
            start_calctime = t.time()
            samples = self.__sdr.read_samples(size * 1024)
            # use matplotlib to estimate the PSD and save the max power
            power, freqs = plt.psd(samples, NFFT=1024,
                Fs=self.__sdr.sample_rate/1e6, Fc=self.__sdr.center_freq/1e6)
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

    def find_peaks(self, power, freqs, interval=80):
        """Find the maximum values of the PSD around the
        frequencies assigned to the object.
        """
        length = len(self.__freq)
        power = power.tolist()
        freqs_temp = freqs.tolist()
        freqs_temp = [float(xx) for xx in freqs]
        freqs_temp = [round(xxx, 1) for xxx in freqs]
        # print "\n Freq Temp: \n"
        # print freqs_temp
        # print "\n Length of Freq vector: \n"
        # print len(freqs)
        pmax = []
        for i in range(length):
            marker = freqs_temp.index(self.__freq[i]/1e6)
            pmax.append(max(power[marker: marker + interval]))
        return pmax

    def rpi_get_power(self, printing=0, size=256):
        """Routine for Raspberry Pi.
        keyword arguments:
        printing -- visible output on terminal (default  0)
        size -- measure for length of fft (default 256)"""
        running = True
        pmax = []
        while running:
            try:
                samples = self.__sdr.read_samples(size * 1024)
                power, freqs = plt.psd(samples, NFFT=1024,
                    Fs=self.__sdr.sample_rate/1e6,
                    Fc=self.__sdr.center_freq/1e6)
                pmax.append(self.find_peaks(power, freqs))
                if printing:
                    print self.find_peaks(power, freqs)
                    print "\n"
                else:
                    pass
            except KeyboardInterrupt:
                print "Process interrupted by user"
                return pmax

def write_to_file(results, text, filename="Experiments"):
    """Save experimental results in a simple text file.
    arguments:
    results -- list
    text -- description of results
    filename -- name of file (default Experiments)
    """
    datei = open(filename, "a")
    datei.write(t.ctime() + "\n")
    datei.write(text + "\n")
    datei.write(str(results))
    datei.write("\n\n")
    datei.close()
