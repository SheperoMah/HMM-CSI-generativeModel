#!/usr/bin/env python3
import sys
import numpy as np
import auxiliaryFuncs as auxFs
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from statsmodels.tsa.stattools import acf

def mean_abs_error(x1, x2):
	return(np.nanmean(np.abs(x1 - x2)))

def main(fileName, ymax, fntSize=16, numLags=60, fgSz=(6,6)):
    ds = np.loadtxt(f"{fileName}.txt").transpose()
    trDs, valDs = auxFs.divide_data(ds)
    lengthTr = auxFs.calculate_length_sequence(trDs)
    lengthVal = auxFs.calculate_length_sequence(valDs)
    trDs = trDs.reshape(-1,1)
    valDs = valDs.reshape(-1,1)
    dataSets = [trDs, valDs]
    lengths = [lengthTr, lengthVal]
        
    sample = np.loadtxt(f"{fileName}_3_100_fixedMeans.txt")
    
    fig = plt.figure(figsize=fgSz)
    plt.rcParams.update({'font.size': fntSize})
    ax = fig.add_subplot(111)
    acfTrnData = acf(trDs, nlags=numLags)
    acfTstData = acf(valDs, nlags=numLags)
    acfSample = auxFs.ACF(sample, numLags)
    ax.plot(acfTrnData, color='k', linestyle='-')
    ax.plot(acfTstData, color='k', linestyle=':')
    ax.plot(acfSample)
    ax.set_xlabel("lags (minute)")
    ax.set_ylabel("ACF")
    ax.set_xlim(0,60)
    
    auxFs.clean_axes(ax)
    ax.legend(["Training data", "Test data", "model sample"])
    fig.savefig(f"{fileName}_acf_fixedMeans.pdf")
    print(mean_abs_error(acfTstData, acfSample)) 

    histFigure = plt.figure(figsize=(6,6))
    ax2 = histFigure.add_subplot(111)
    ax2.hist(valDs, 100, density=True)
    def do_on_axes(ax):
        ax.set_xlabel("CSI")
        ax.set_ylabel("PDF")
        auxFs.clean_axes(ax)
        ax.set_xlim(0,1.75)
        ax.set_ylim(top=ymax)
        #ax.set_xticklabels([])
        #ax.set_yticklabels([])
    do_on_axes(ax2)
    histFigure.savefig(f"{fileName}_histogram_fixedMeans_test.pdf") 

    hist2 = plt.figure(figsize=(6,6))
    ax2 = hist2.add_subplot(111)
    ax2.hist(sample, 100, density=True)
    do_on_axes(ax2)
    hist2.savefig(f"{fileName}_histogram_fixedMeans_sample.pdf") 

if __name__ == "__main__":
    fileName = sys.argv[1]
    ymax = float(sys.argv[2])
    main(fileName, ymax)
