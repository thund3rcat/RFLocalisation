# import modules
import time as t
import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import *

# create sdr object
sdr = RtlSdr()

def psdplot(f):

    # configure device
    sdr.sample_rate = 2.4e6
    sdr.center_freq = f
    sdr.gain = 4
    # plt.axis([433.5, 436.5, -50, 30])
    plt.ion()
    #plt.figure()
    plt.show()

    while True:
        pass  # Busy-wait for keyboard interrupt (Ctrl+C)
        plt.clf()
        plt.axis([433.5, 436.5, -50, 30])
        samples = sdr.read_samples(256 * 1024)
        # use matplotlib to estimate and plot the PSD
        plt.psd(samples, NFFT=1024, Fs=sdr.sample_rate / 1e6, Fc=sdr.center_freq / 1e6)
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Relative power (dB)")
        plt.draw()
        t.sleep(0.05)


