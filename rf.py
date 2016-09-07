"""This module hacks a DVBT-dongle and abuses it
as a spectrum analyzer between 23 an 1,700 MHz.
"""

# import modules
from abc import ABCMeta, abstractmethod
import time as t
import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import *
from scipy import signal
from scipy.optimize import curve_fit
from scipy.special import lambertw
from matplotlib2tikz import save as tikz_save
import os
import py2tikz

sdr = RtlSdr()
sdr.gain = 1
sdr.sample_rate = 2.048e6

class RfEar(object):
    """A simple class to compute PSD with a DVBT-dongle."""

    __metaclass__ = ABCMeta

    def __init__(self, *args):
        """Keyword-arguments:
        *args -- list of frequencies (must be in a range of __sdr.sample_rate)
        """
        self.__freq = []
        self.set_freq(self, *args)
        self.__size = 0
        self.set_size(256)
        self.__d0 = np.array(0, dtype=float)
        self.__p0 = np.array(0, dtype=float)

    def set_size(self, size):
        self.__size = size

    def get_size(self):
        return self.__size

    def get_iq(self):
        """ Read iq samples at a certain frequency 
        """
        samples = sdr.read_samples(self.__size * 1024)
        return samples

    def set_freq(self, *args):
        """Defines frequencies where to listen (between 27MMz and 1.7GHz).
        The range must be in the __sdr.sample_rate bandwidth.
        """
        self.__freq[:] = []
        for arg in args:
            self.__freq.append(arg)
        if not isinstance(self.__freq[0], float):
            self.__freq = self.__freq[1:]
        sdr.center_freq = np.mean(self.__freq)

    def get_freq(self):
        """Returns list of frequencies assigned to the object."""
        return self.__freq

    def set_srate(self, srate):
        """Defines the sampling rate (default 2.4e6)."""
        sdr.sample_rate = srate

    def get_srate(self):
        """Returns sample rate assigned to object
        and gives default tuner value.
        range is between 1.0 and 3.2 MHz
        """
        print 'Default sample rate: 2.4MHz'
        print 'Current sample rate: '
        return sdr.sample_rate

    def plot_psd(self):
        """Get Power Spectral Density Live Plot.
        """
        plt.ion()      # turn interactive mode on
        drawing = True
        while drawing:
            try:
                # Busy-wait for keyboard interrupt (Ctrl+C)
                plt.clf()
                #plt.axis([self.__sdr.center_freq/1e6-.5*self.__sdr.sample_rate/1e6,
                #   self.__sdr.center_freq/1e6+.5*self.__sdr.sample_rate/1e6, -120, 20])
                plt.axis([-1.5e6,
                  1.5e6, -120, 0])
                samples = self.get_iq()
                # use matplotlib to estimate and plot the PSD
                #freq, pxx_den = plt.psd(samples, NFFT=1024, Fs=self.__sdr.sample_rate,
                #    Fc=self.__sdr.center_freq/1e6)
                freq, pxx_den = signal.periodogram(samples,
                   fs=sdr.sample_rate, nfft=1024)
                #pxx_den_1 = pxx_den[:len(pxx_den)/2]
                #pxx_den_2 = pxx_den[len(pxx_den)/2:]
                #pxx_den = pxx_den_2+pxx_den_1
                #freq = np.linspace(self.__sdr.center_freq/1e6-.5*self.__sdr.sample_rate/1e6,
                #    self.__sdr.center_freq/1e6+.5*self.__sdr.sample_rate/1e6,
                #    len(pxx_den))
                plt.plot(freq, 10*np.log10(pxx_den)) #rss in dB
                plt.grid()
                plt.xlabel('Frequency [MHz]')
                plt.ylabel('Power [dB]')
                plt.show()
                plt.pause(0.001)
            except KeyboardInterrupt:
                plt.show()
                print 'Liveplot interrupted by user'
                drawing = False
        return pxx_den, freq

    def get_rss(self):
        """Find maximum power values around specified freq points.
        returns power in received signal strength (rss) in dB
        """
        samples = self.get_iq()
        freq, pxx_den = signal.periodogram(samples,
            fs=sdr.sample_rate, nfft=1024)
        del freq
        if len(self.__freq) == 1:
            rss = [10*np.log10(max(pxx_den))] #Power in dB !!!
        elif len(self.__freq) == 2:
            pxx_den_left = pxx_den[:len(pxx_den)/2]
            pxx_den_right = pxx_den[len(pxx_den)/2:]
            rss = [10*np.log10(max(pxx_den_left)),
                10*np.log10(max(pxx_den_right))]
        return rss

    @abstractmethod
    def rfear_type(self):
        """"Return a string representing the type of rfear this is."""
        pass

    #unused
    def rpi_get_power(self, printing=0, size=256):
        """Routine for Raspberry Pi.
        Keyword arguments:
        printing -- visible output on terminal (default  0)
        size -- measure for length of fft (default 256*1024)
        """
        running = True
        pmax = []
        self.set_size(size)
        while running:
            try:
                pmax.append(self.get_rss())
                if printing:
                    print self.get_rss()
                    print '\n'
                else:
                    pass
            except KeyboardInterrupt:
                print 'Process interrupted by user'
                return pmax

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

def plot_map(pos_est, x_max=53.5):
    """ plot path from recorded data
    """
    x_min = 0
    y_min = -100
    y_max = 100
    x_est = []
    y_est = []
    for i in range(len(pos_est)):
        x_est.append((np.power(pos_est[i][0],2)-np.power(pos_est[i][1],2)+np.power(x_max,2))/(2*x_max))
        y_est.append(np.sqrt(np.power(pos_est[i][0],2) - np.power(x_est[-1],2)))
    plt.figure()
    plt.plot(x_est, y_est, 'bo')
    plt.axis([x_min, x_max, y_min, y_max])
    plt.grid()
    plt.show()
    #return x_est, y_est

def tikz_make(filename, f_h='4cm', f_w='6cm'):
    """Moves matplotlib2tikz made tex file to Tikz folder
    """
    tikz_save(filename, figureheight=f_h, figurewidth=f_w)
    os.system('mv ' + filename +' /home/michael/latex/PA/Tikz')

class CalEar(RfEar):

    def __init__(self, *args):
        RfEar.__init__(self, *args)

    def plot_rss(self, time=10):
        """Measure power values of specified frequency
        for a certain time and performance.
        Keyword arguments:
        time -- time of measurement in seconds (default  10)
        """
        powerstack = []
        elapsed_time = 0
        timestack = []
        while elapsed_time < time:
            start_calctime = t.time()
            # use matplotlib to estimate the PSD and save the max power
            powerstack.append(self.get_rss())
            t.sleep(0.005)
            calctime = t.time() - start_calctime
            timestack.append(calctime)
            elapsed_time = elapsed_time + calctime
            #print "Finished"
            calctime = np.mean(calctime)   
        #text = "Rechenzeit pro Update: " + calctime + " sec\n Gesamtzeit: " + elapsed_time + " sec"
        print 'Number of samples per update:\n'
        print self.get_size()*1024
        print '\n'
        print 'Calculation time per update (sec):\n'
        print calctime
        print '\n'
        print 'Total time (sec):\n'
        print elapsed_time
        plt.clf()
        plt.grid()
        plt.axis([0, len(powerstack), -120, 10])
        plt.plot(powerstack, 'o')
        plt.xlabel('Updates')
        plt.ylabel('Maximum power (dB)')
        plt.figure()
        plt.plot(timestack, 'g^')
        plt.xlabel('Updates')
        plt.ylabel('Computation time (sec)')
        plt.show()
        return powerstack

    def make_test(self, time=10):
        """Interactive method to get PSD data
        at characteristic frequencies.
        Keyword arguments:
        time -- time of measurement in sec (default 10s)
        """
        testing = True
        modeldata = []
        variance = []
        plt.figure()
        plt.grid()
        while testing:
            try:
                raw_input('Press Enter to make a measurement,'
                    ' or Ctrl+C+Enter to stop testing:\n')
                elapsed_time = 0
                powerstack = []
                print ' ... measuring ...'
                while elapsed_time < time:
                    start_calctime = t.time()
                    powerstack.append(self.get_rss())
                    calc_time = t.time() - start_calctime
                    elapsed_time = elapsed_time + calc_time
                    t.sleep(0.01)
                print 'done\n'
                t.sleep(0.5)
                print ' ... evaluating ...'
                modeldata.append(np.mean(powerstack))
                variance.append(np.var(powerstack))
                plt.clf()
                plt.errorbar(range(len(modeldata)), modeldata, yerr=variance,
                fmt='o', ecolor='g')
                plt.xlabel('Evaluations')
                plt.ylabel('Mean Maximum power [dB] default')
                plt.show()
                del powerstack
                print 'done\n'
                t.sleep(0.5)
            except KeyboardInterrupt:
                print 'Testing finished'
                testing = False
        return modeldata, variance

    def get_performance(self, bandwidth=2.4e6):
        """Measure performance at certain sizes and sampling rates
        keyword arguments:
        bandwidth -- sampling rate of dvbt-dongle
        """
        self.set_srate(bandwidth)
        measurements = 200
        SIZE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 46, 52, 58, 64, 76, 88, 100, 128, 160, 192, 234, 256]
        SIZE = [4, 8, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256]
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
                self.set_size(i)
                powerstack.append(self.get_rss())
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
        plt.errorbar(SIZE, MEAN, yerr=VAR,
                fmt='o', ecolor='g')
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

    def get_model(self, pdata, vdata):
        """Create a function to fit with measured data.
        alpha and xi are the coefficients that curve_fit will calculate.
        The function structure is known and some coeffs may be known
        through calibration.
        Keyword arguments:
        pdata -- array containing the power values in dB
        vdata -- array containing the variance of the measurements
        """
        x_init = raw_input('Please enter initial distance [cm]: ')
        x_step = raw_input('Please enter step size [cm]:')
        xdata = np.arange(int(x_init), int(x_init)+len(pdata)*int(x_step), int(x_step))
        xdata = np.array(xdata, dtype=float)
        pdata = np.array(pdata, dtype=float)
        vdata = np.array(vdata, dtype=float)
        plt.figure()
        plt.grid()
        plt.errorbar(xdata, pdata, yerr=vdata,
                fmt='ro', ecolor='g', label='Original Data')
        def func(xdata, alpha, xi):
            #d_ref, p_ref = self.get_caldata()
            #return p_ref-10*alpha*np.log10(xdata/d_ref)+xi
            return -20*np.log10(xdata)-alpha*xdata-xi
        popt, pcov = curve_fit(func, xdata, pdata)
        del pcov
        print 'alpha = %s , xi = %s' % (popt[0], popt[1])
        xdata = np.linspace(xdata[0], xdata[-1], num=1000)
        plt.plot(xdata, func(xdata, *popt), label='Fitted Curve')
        plt.legend(loc='upper right')
        plt.xlabel('Distance [cm]')
        plt.ylabel('RSS [dB]')
        plt.show()
        return popt

    def rfear_type(self):
        """"Return a string representing the type of rfear this is."""
        return 'CalEar'

class LocEar(RfEar):

    def __init__(self, alpha, xi, *args):
        RfEar.__init__(self, *args)
        self.__alpha = alpha
        self.__xi = xi
    '''
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
    '''
    def map_path(self, popt, d_t=55):
        """Maps estimated location
        Keyword argumennts:
        popt -- model parameter
        """
        x_min = -10
        x_max = d_t+10
        y_min = -100
        y_max = 100
        plt.axis([x_min, x_max, y_min, y_max])
        plt.ion()
        plt.grid()
        plt.xlabel('x-Axis [cm]')
        plt.ylabel('y-Axis [cm]')
        drawing = True
        pos_est = []
        try:
            while drawing:
                rss = self.get_rss()
                pos_est.append(lambertloc(rss, popt[0], popt[1]))
                if len(pos_est[-1]) == 1:
                    plt.plot(pos_est[-1], 0, 'bo')
                elif len(pos_est[-1]) == 2:
                    x_est = (pos_est[-1][0]**2-pos_est[-1][1]**2+d_t**2)/(2*d_t)
                    y_est = np.sqrt(pos_est[-1][0]**2 - x_est**2)
                    print [x_est, y_est]
                    plt.plot(x_est, y_est, 'bo')
                plt.show()
                plt.pause(0.001)
                print pos_est[-1]
                print '\n'
        except KeyboardInterrupt:
            print 'Localization interrupted by user'
            drawing = False
        return pos_est

    def lambertloc(self, rss):
        """Inverse function of the range sensor model
        Keyword arguments:
        rss -- received power values
        alpha, xi -- coeffs of model estimation
        """
        Z = [20/(np.log(10)*self.__alpha)*lambertw(np.log(10)*self.__alpha/20*np.exp(-np.log(10)/20*(i+self.__xi))) for i in rss]
        return [z.real for z in Z]

    def rfear_type(self):
        """"Return a string representing the type of rfear this is."""
        return 'LocEar'


