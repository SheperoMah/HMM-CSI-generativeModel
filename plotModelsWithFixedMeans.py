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
    acfData = acf(valDs, nlags=numLags)
    acfSample = acf(sample[0:10000], nlags=numLags)
    ax.plot(acfData)
    ax.plot(acfSample)
    ax.set_xlabel("lags (minute)")
    ax.set_ylabel("ACF")
    ax.set_xlim(0,60)
    
    auxFs.clean_axes(ax)
    ax.legend(["Test data", "model sample"])
    fig.savefig(f"{fileName}_acf_fixedMeans.pdf")
    print(mean_abs_error(acfData, acfSample)) 

    histFigure = plt.figure(figsize=(12,6))
    ax2 = histFigure.add_subplot(121)
    ax2.hist(valDs, 100, density=True)
    ax3 = histFigure.add_subplot(122)
    ax3.hist(sample, 100, density=True)
    ax2.set_title("(a)")
    ax3.set_title("(b)")
    for i in [ax2, ax3]:
        i.set_xlabel("CSI")
        i.set_ylabel("pdf")
        auxFs.clean_axes(i)
        i.set_xlim(0,1.75)
        i.set_ylim(top=ymax)
        i.set_yticklabels([])
        i.set_xticklabels([])
    histFigure.savefig(f"{fileName}_histogram_fixedMeans.pdf") 

if __name__ == "__main__":
    fileName = sys.argv[1]
    ymax = float(sys.argv[2])
    main(fileName, ymax)
