#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import auxiliaryFuncs as auxFs

def main(fileName, ymax=3.5, fntSize=14):
	nbins = 100
	seed = 100

	ds = np.loadtxt(f"{fileName}.txt").transpose()
	trDs, valDs = auxFs.divide_data(ds)
	data = valDs.flatten(order="C")
	lengthdata = data.shape[0]
	plt.rcParams.update({'font.size': fntSize})


	# Plot histogram
	fig = plt.figure(figsize=(6,8.5))
	ax = plt.subplot(4,3,1)
	ax.hist(data, nbins, density=True)
	ax.set_ylabel("pdf")
	#ax.set_xlabel("CSI")
	ax.set_title("Test data")
	ax.set_xlim(0,1.75)
	ax.set_ylim(0,ymax)
	ax.set_yticklabels([])
	ax.set_xticklabels([])
	auxFs.clean_axes(ax)

	for i in range(2,13):
		fn = f"{fileName}_{i}_100.txt"
		X = np.loadtxt(fn)
		#fig = plt.figure(figsize=(6,6))
		ax = plt.subplot(4,3,i)
		ax.hist(X, nbins, density=True)
		if i in [4,7,10]:
			ax.set_ylabel("pdf")
		if i in [10, 11, 12]:
			ax.set_xlabel("CSI")
		ax.set_title(f"n = {i}")
		#ax.set_xlabel("CSI")
		auxFs.clean_axes(ax)
		ax.set_xlim(0,1.75)
		ax.set_ylim(0,ymax)
		ax.set_yticklabels([])
		ax.set_xticklabels([])
	fig.savefig(f"{fileName}_test_histogram.pdf")



if __name__ == "__main__":
	fileName = sys.argv[1]
	ymax = float(sys.argv[2])
	main(fileName, ymax)
