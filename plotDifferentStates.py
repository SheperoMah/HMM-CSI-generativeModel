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

def main(fileName, fntSize = 12):
	numLags = 60
	ds = np.loadtxt(f"{fileName}.txt").transpose()
	trDs, valDs = auxFs.divide_data(ds)
	data = valDs.flatten(order="C")
	lengthdata = data.shape[0]
	plt.rcParams.update({'font.size': fntSize})
	cols = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
	lnstyle = ['-',':', '-.', '--']
	fig1 = plt.figure(figsize=(6,6))
	fig2 = plt.figure(figsize=(6,6))
	ax1 = fig1.add_subplot(111)
	ax2 = fig2.add_subplot(111)
	xKde = np.linspace(0.0, 1.5, num=50)
	ax1.plot(xKde, kde_fit_estimate(data, xKde), 'k')
	acfTst = acf(data, nlags=numLags)
	maeDict = {}
	ax2.plot(acfTst, 'k')
	filenames = [f"{fileName}_{i}_100.pkl" for i in range(2,13)]
	for i,v in enumerate(filenames):
		model = joblib.load(v)
		X,_ = model.sample(lengthdata)
		ax1.plot(xKde, kde_fit_estimate(X.flatten(), xKde), linestyle=lnstyle[i%3], color=cols[i%10])
		acfModel = acf(X, nlags=numLags)
		ax2.plot(acfModel, linestyle=lnstyle[i%3], color=cols[i%10])
		maeDict[str(i+2)] = mean_abs_error(acfTst, acfModel)

	legendTxt = ["Test data"] + [f"n = {i}"for i in range(2,13)]
	ax1.legend(legendTxt)
	ax1.set_xlim(0,1.75)
	ax1.set_ylim(0,3.5)
	ax1.set_ylabel("pdf")
	ax1.set_xlabel("CSI")
	auxFs.clean_axes(ax1)
	ax2.legend(legendTxt)
	ax2.set_xlim(0,60)
	ax2.set_ylabel("ACF")
	ax2.set_xlabel("lags (minute)")
	auxFs.clean_axes(ax2)
	fig1.savefig(f"{fileName}_states_histograms.pdf")
	fig2.savefig(f"{fileName}_states_acf.pdf")
	with open(f"{fileName}_acf_mae.json", 'w') as f:
		json.dump(maeDict, f, indent=4)

if __name__ == "__main__":

	fn = sys.argv[1]
	main(fn)