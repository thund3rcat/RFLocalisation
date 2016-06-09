"""This module hacks a DVBT-dongle and abuses it
as a spectrum analyzer between 23 an 1,700 MHz.
"""

# import modules
# import modules
import time as t
import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import *
from scipy import signal
from scipy.optimize import curve_fit

class RFear(object):
    """A simple class to compute PSD with a DVBT-dongle."""
    def __init__(self, *args):
        """Keyword-arguments:
        gain -- run rtl_test in the console to see available gains
        *args -- list of frequencies (must be in a range of __sdr.sample_rate)
        """
        self.__sdr = RtlSdr()
        self.__sdr.sample_rate = 2.4e6
        self.__sdr.gain = 1
        self.__freq = []
        self.set_freq(self, *args)
        self.__d0 = np.array(0, dtype=float)
        self.__p0 = np.array(0, dtype=float)

    def __del__(self):
        del self.__sdr

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

    def get_freq(self):
        """Returns list of frequencies assigned to the object."""
        return self.__freq

    def set_gain(self, gain):
        """Defines the gain which amlifies the recieved signal.
        Run rtl_test to get valid gain values.
        """
        self.__sdr.gain = gain

    def get_gain(self):
        """Returns gain assigned to object and lists valid gains"""
        print 'Valid gains:'
        print [x/10.0 for x in self.__sdr.gain_values]
        return self.__sdr.gain

    def set_srate(self, srate):
        """Defines the sampling rate (default 2.4e6)."""
        self.__sdr.sample_rate = srate

    def get_srate(self):
        """Returns sample rate assigned to object
        and gives default tuner value.
        range is between 1.0 and 3.2 MHz
        """
        print 'Default sample rate: 2.4MHz'
        print 'Current sample rate: '
        return self.__sdr.sample_rate

    def calibrate(self):
        """Set reference values P0 and d0"""
        d_ref = raw_input('Please enter distance'
            'from transmitter to receiver [cm]: ')
        self.__d0 = np.array(d_ref, dtype=float)
        p_ref, variance = self.make_test(30)
        del variance
        self.__p0 = np.array(p_ref, dtype=float)

    def get_caldata(self):
        """Returns the reference values from calibration"""
        return self.__d0, self.__p0

    def get_psd(self, size=256):
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
                samples = self.__sdr.read_samples(size * 1024)
                # use matplotlib to estimate and plot the PSD
                #plt.psd(samples, NFFT=1024, Fs=self.__sdr.sample_rate/1e6,
                #    Fc=self.__sdr.center_freq/1e6)
                freq, pxx_den = signal.periodogram(samples,
                    fs=self.__sdr.sample_rate/1e6, nfft=1024)
                plt.plot(freq+self.__sdr.center_freq/1e6, 10*np.log10(pxx_den))
                plt.grid()
                plt.xlabel('Frequency (MHz)')
                plt.ylabel('Relative power (dBm)')
                plt.draw()
                t.sleep(0.01)
            except KeyboardInterrupt:
                plt.draw()
                print 'Liveplot interrupted by user'
                drawing = False

    def get_power(self, time=10, size=256):
        """Measure power values of specified frequency
        for a certain time and performance.
        keyword arguments:
        time -- time of measurement in seconds (default  10)
        size -- measure for length of fft (default 256)
        """
        powerstack = []
        elapsed_time = 0
        timestack = []
        while elapsed_time < time:
            start_calctime = t.time()
            # use matplotlib to estimate the PSD and save the max power
            powerstack.append(self.find_peaks2(size))
            t.sleep(0.005)
            calctime = t.time() - start_calctime
            timestack.append(calctime)
            elapsed_time = elapsed_time + calctime
            #print "Finished"
            calctime = np.mean(calctime)   
        #text = "Rechenzeit pro Update: " + calctime + " sec\n Gesamtzeit: " + elapsed_time + " sec"
        print 'Number of samples per update:\n'
        print size*1024
        print '\n'
        print 'Calculation time per update (sec):\n'
        print calctime
        print '\n'
        print 'Total time (sec):\n'
        print elapsed_time
        plt.clf()
        plt.grid()
        plt.axis([0, len(powerstack), -50, 30])
        plt.plot(10*np.log10(powerstack), 'o')
        plt.xlabel('Updates')
        plt.ylabel('Maximum power (dBm)')
        plt.figure()
        plt.plot(timestack, 'g^')
        plt.xlabel('Updates')
        plt.ylabel('Computation time (sec)')
        plt.show()
        return powerstack

    def find_peaks(self, size=256, interval=100, glb=False):
        """Find the maximum values of the PSD around the
        frequencies assigned to the object.
        glb = True: find global PSD and respective frequency
        in a range from 27 to 1700MHz
        """
        pmax = []
        freqmax = []
        if not glb:
            samples = self.__sdr.read_samples(size * 1024)
            power, freqs = plt.psd(samples, NFFT=1024,
                Fs=self.__sdr.sample_rate/1e6,
                Fc=self.__sdr.center_freq/1e6)
            length = len(self.__freq)
            power = power.tolist()
            freqs_temp = freqs.tolist()
            freqs_temp = [float(xx) for xx in freqs]
            freqs_temp = [round(xxx, 1) for xxx in freqs]
            # print "\n Freq Temp: \n"
            # print freqs_temp
            # print "\n Length of Freq vector: \n"
            # print len(freqs)
            for i in range(length):
                marker = freqs_temp.index(self.__freq[i]/1e6)
                pmax.append(max(power[marker-int(.1*interval):marker
                    +int(.9*interval)]))
            return pmax
        else:
            freq_range = np.arange(24e6, 1702e6, 2e6)
            for i in freq_range:
                self.set_freq(i)
                samples = self.__sdr.read_samples(size * 1024)
                power, freqs = plt.psd(samples, NFFT=1024,
                    Fs=self.__sdr.sample_rate/1e6,
                    Fc=self.__sdr.center_freq/1e6)
                pmax.append(max(power))
                freqmax.append(freqs[pmax.index(max(power))])
                plt.close('all')
                print i
            print max(pmax)
            print 'at'
            print freqmax[pmax.index(max(pmax))]
            return max(pmax), freqmax[pmax.index(max(pmax))]
        
    def find_peaks2(self, size=256):
        #length = len(self.__freq)
        pmax = []
        samples = self.__sdr.read_samples(size * 1024)
        freq, pxx_den = signal.periodogram(samples,
            fs=self.__sdr.sample_rate/1e6, nfft=1024)
        pmax.append(max(pxx_den))
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
                pmax.append(self.find_peaks2(size))
                if printing:
                    print self.find_peaks2(size)
                    print '\n'
                else:
                    pass
            except KeyboardInterrupt:
                print 'Process interrupted by user'
                return pmax

    def make_test(self, time=10, size=256):
        """Interactive method to get PSD data
        at characteristic frequencies.
        """
        testing = True
        modeldata = []
        variance = []
        while testing:
            try:
                raw_input('Press Enter to make a measurement,'
                    ' or Ctrl+C+Enter to stop testing:\n')
                elapsed_time = 0
                powerstack = []
                print ' ... measuring ...'
                while elapsed_time < time:
                    start_calctime = t.time()
                    powerstack.append(self.find_peaks2(size))
                    calc_time = t.time() - start_calctime
                    elapsed_time = elapsed_time + calc_time
                    t.sleep(0.01)
                print 'done\n'
                t.sleep(0.5)
                print ' ... evaluating ...'
                modeldata.append(np.mean(powerstack))
                variance.append(np.var(powerstack))
                del powerstack
                print 'done\n'
                t.sleep(0.5)
            except KeyboardInterrupt:
                print 'Testing finished'
                testing = False
        plt.figure()
        plt.grid()
        #plt.axis([0, len(modeldata), -50, 30])
        plt.errorbar(range(len(modeldata)), modeldata, yerr=variance,
            fmt='o', ecolor='g')
        plt.xlabel('Evaluations')
        plt.ylabel('Mean Maximum power [?]')
        return modeldata, variance

    def get_performance(self, bandwidth=2.4e6):
        """Measure performance at certain sizes and sampling rates
        keyword arguments:
        time -- time of measurement in seconds (default  10)
        size -- measure for length of fft (default 256)
        bandwidth -- sampling rate of dvbt-dongle
        """
        self.set_srate(bandwidth)
        measurements = 200
        SIZE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 46, 52, 58, 64, 76, 88, 100, 128, 160, 192, 234, 256]
        VAR = []
        MEAN = []
        UPDATE = []  
        total_time = 0     
        for i in SIZE:
            cnt = 0
            powerstack = []
            timestack = []
            elapsed_time = 0
            while cnt <= measurements:
                cnt = cnt+1
                start_calctime = t.time()
                # use matplotlib to estimate the PSD and save the max power
                powerstack.append(self.find_peaks2(i))
                t.sleep(0.005)
                calctime = t.time() - start_calctime
                timestack.append(calctime)
                elapsed_time = elapsed_time + calctime
            calctime = np.mean(timestack)
            VAR.append(np.var(powerstack))
            MEAN.append(np.mean(powerstack))
            UPDATE.append(calctime)
            total_time = total_time+elapsed_time
        print 'Finished.'
        print 'Total time [sec]: '
        print total_time
        plt.figure()
        plt.grid()
        plt.plot(SIZE, VAR, 'ro')
        plt.xlabel('Sample Size (*1024)')
        plt.ylabel('Variance (dB)')
        plt.figure()
        plt.grid()
        plt.plot(SIZE, MEAN, 'x')
        plt.xlabel('Sample Size (*1024)')
        plt.ylabel('Mean Value (dB)')
        plt.figure()
        plt.grid()
        plt.plot(SIZE, UPDATE, 'g^')
        plt.xlabel('Sample Size (*1024)')
        plt.ylabel('Update rate (sec)')
        plt.show()
        return SIZE, VAR, MEAN, UPDATE

    def get_model(self, ddata, pdata, vdata):
        """Create a function to fit with measured data.
        alpha and xi are the coefficients that curve_fit will calculate.
        The function structure is known and some coeffs are known
        through calibration.
        """
        ddata = np.array(ddata, dtype=float)
        pdata = np.array(pdata, dtype=float)
        vdata = np.array(vdata, dtype=float)
        plt.grid()
        plt.errorbar(ddata, pdata, yerr=vdata,
                fmt='ro', ecolor='g', label='Original Data')
        def func(ddata, alpha, xi):
            d_ref, p_ref = self.get_caldata()
            return p_ref-10*alpha*np.log10(ddata/d_ref)+xi
        popt, pcov = curve_fit(func, ddata, pdata)
        del pcov
        print 'alpha = %s , xi = %s' % (popt[0], popt[1])
        ddata = np.linspace(ddata[0], ddata[-1], num=1000)
        plt.plot(ddata,func(ddata, *popt), label='Fitted Curve')
        plt.legend(loc='upper right')
        plt.show()
        return popt


def plot_result(results):
    """Plot results extracted from textfile"""
    plt.figure()
    plt.grid()
    plt.axis([0, len(results), -50, 30])
    plt.plot(10*np.log10(results), 'o')
    plt.xlabel('Updates')
    plt.ylabel('Maximum power (dBm)')
    plt.show()

def write_to_file(results, text, filename='Experiments'):
    """Save experimental results in a simple text file.
    arguments:
    results -- list
    text -- description of results
    filename -- name of file (default Experiments)
    """
    datei = open(filename, 'a')
    datei.write(t.ctime() + '\n')
    datei.write(text + '\n')
    datei.write(str(results))
    datei.write('\n\n')
    datei.close()


