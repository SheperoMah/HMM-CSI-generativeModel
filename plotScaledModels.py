#!/usr/bin/env python3


import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from statsmodels.tsa.stattools import acf
import auxiliaryFuncs as auxFs
from scipy.stats import gaussian_kde
import json


def kde_fit_estimate(data, x):
    kde = gaussian_kde(data)
    return(kde.evaluate(x))

def mean_abs_error(x1, x2):
    return(np.mean(abs(x1 - x2)))

def main(fileName, fntSize=16):
    numLags = 60
    ds = np.loadtxt(f"{fileName}.txt").transpose()
    trDs, valDs = auxFs.divide_data(ds)
    data = valDs.flatten(order="C")
    lengthdata = data.shape[0]
    plt.rcParams.update({'font.size': fntSize})
    cols = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    lnstyle = ['-',':', '-.', '--']
    fig1 = plt.figure(figsize=(8,8))
    fig2 = plt.figure(figsize=(8,8))
    ax1 = fig1.add_subplot(111, position=[0.175, 0.1, 0.65, 0.65])
    ax2 = fig2.add_subplot(111, position=[0.175, 0.1, 0.65, 0.65])
    xKde = np.linspace(0.0, 1.5, num=150)
    ax1.plot(xKde, kde_fit_estimate(data, xKde), 'k')
    acfTst = acf(data, nlags=numLags)
    maeDict = {}
    ax2.plot(acf(trDs.flatten(order="c"), nlags=numLags), color='k', linestyle='-')
    ax2.plot(acfTst, color='k',linestyle=':')
    scales = [20, 30, 40, 50, 60, 80, 100]
    filenames = [f"{fileName}_3_{i}_fixedMeans.txt" for i in scales]
    for i,v in enumerate(filenames):
        X = np.loadtxt(v)
        ax1.plot(xKde, kde_fit_estimate(X.flatten(), xKde), linestyle=lnstyle[i%3], color=cols[i%10])
        acfModel = auxFs.ACF(X.flatten(), numLags)
        ax2.plot(acfModel, linestyle=lnstyle[i%3], color=cols[i%10])
        maeDict[str(scales[i]/100)] = mean_abs_error(acfTst, acfModel)

    legendTxt = ["Test data"] + [r"$\phi$" + f" = {i/100:0.2f}"for i in scales]
    ax1.legend(legendTxt, loc='lower center',                                  
            bbox_to_anchor=(0.5, 1.01), ncol=4)
    ax1.set_xlim(0,1.75)
    ax1.set_ylim(0,4.0)
    ax1.set_ylabel("PDF")
    ax1.set_xlabel("CSI")
    auxFs.clean_axes(ax1)
    ax2.legend(["Training data"] + legendTxt, loc='lower center',                                  bbox_to_anchor=(0.5, 1.01), ncol=3)
    ax2.set_xlim(0,60)
    ax2.set_ylabel("ACF")
    ax2.set_xlabel("lags (minute)")
    auxFs.clean_axes(ax2)
    fig1.savefig(f"{fileName}_scaled_histograms.pdf")
    fig2.savefig(f"{fileName}_scaled_acf.pdf")
    with open(f"{fileName}_acf_mae_scaled.json", 'w') as f:
        json.dump(maeDict, f, indent=4)

if __name__ == "__main__":
    fileName = sys.argv[1]
    main(fileName)
