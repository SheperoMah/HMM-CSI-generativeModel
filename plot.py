#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from statsmodels.tsa.stattools import acf

def main(filename):
	data = np.loadtxt(f"{filename}.txt")
	data = data.flatten(order="F")
	plt.rcParams.update({'font.size': 12})
	fig1 = plt.figure(figsize=(6,6))
	fig2 = plt.figure(figsize=(6,6))
	ax1 = fig1.add_subplot(111)
	ax2 = fig2.add_subplot(111)
	ax1.hist(data, 100, alpha=0.4, density=True)
	ax2.plot(acf(data, nlags=120))
	k = [100, 80, 40, 30, 20]
	filenames = [f"{filename}_3_{i}.pkl" for i in k]
	for i in filenames:
		model = joblib.load(i)
		X,_ = model.sample(365*120)
		ax1.hist(X, 100, alpha=0.4, density=True)
		ax2.plot(acf(X, nlags=120))

	legendTxt = ["All data"] + [r"$\phi$ " f"= {i/100:0.2f}"for i in k]
	ax1.legend(legendTxt)
	ax1.set_xlim(0,1.75)
	ax1.set_ylabel("pdf")
	ax1.set_xlabel("CSI")
	ax2.legend(legendTxt)
	ax2.set_xlim(0,125)
	ax2.set_ylabel("ACF")
	ax2.set_xlabel("lags (minute)")
	fig1.savefig(f"{filename}_histograms.pdf")
	fig2.savefig(f"{filename}_acf.pdf")


if __name__ == "__main__":
	filename = sys.argv[1]
	main(filename)

