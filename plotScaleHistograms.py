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
	ax = plt.subplot(2,4,1)
	ax.hist(data, nbins, density=True)
	ax.set_ylabel("pdf")
	#ax.set_xlabel("CSI")
	ax.set_title("Test data")
	ax.set_xlim(0,1.75)
	ax.set_ylim(0,ymax)
	ax.set_yticklabels([])
	ax.set_xticklabels([])
	auxFs.clean_axes(ax)

	scales = [20, 30, 40, 50, 60, 80, 100]
	filenames = [f"{fileName}_3_{i}.pkl" for i in scales]
	for i, v in enumerate(filenames):
		model = joblib.load(v)
		X,_ = model.sample(lengthdata)
		#fig = plt.figure(figsize=(6,6))
		ax = plt.subplot(2,4, i+2)
		ax.hist(X, nbins, density=True)
		if i in [3]:
			ax.set_ylabel("pdf")
		if i in range(3,8):
			ax.set_xlabel("CSI")
		ax.set_title(r"$\phi$ = "+ f"{scales[i]/100:.2f}")
		#ax.set_xlabel("CSI")
		auxFs.clean_axes(ax)
		ax.set_xlim(0,1.75)
		ax.set_ylim(0,ymax)
		ax.set_yticklabels([])
		ax.set_xticklabels([])
	fig.savefig(f"{fileName}_scaled_process_histogram.pdf")



if __name__ == "__main__":
	fileName = sys.argv[1]
	ymax = float(sys.argv[2])
	main(fileName, ymax)
