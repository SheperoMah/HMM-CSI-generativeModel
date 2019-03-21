#!/usr/bin/env python3

from statsmodels.tsa.stattools import acf
import numpy as np
import matplotlib.pyplot as plt
import auxiliaryFuncs as auxFs

def main(fntSize=18):
    plt.rcParams.update({'font.size': fntSize})
    numLags = 60
    length = 9000
    mu = [0.433, 0.982, 1.025]
    variance = [0.0312, 0.00032, 0.016]

    samples = [np.random.normal(mu[i], np.sqrt(variance[i]), int(length/3)) for i in range(3)]
    series1 = np.concatenate(samples)
    #series1.reshape(3,-1).flatten(order='F') # flatten row wise
    fig = plt.figure(figsize=(10.5,10.5*9/16))
    ax1 = fig.add_subplot(111)
    ax1.plot(acf(series1, nlags=numLags))
    np.random.shuffle(series1)
    ax1.plot(acf(series1, nlags=numLags))
    ax1.set_xlim(0.0)
    ax1.set_ylabel("ACF")
    ax1.set_xlabel("lags")
    auxFs.clean_axes(ax1)
    ax1.legend(["Process 1", "Process 2"])
    fig.savefig(f"ACF_twoProcesses.pdf")

if __name__ == "__main__":
  main()
