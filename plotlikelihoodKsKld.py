#!/usr/bin/env python3


import sys
import json
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import auxiliaryFuncs as auxFs
import matplotlib.ticker as ticker

def main(filename, seed):
	fgsz = (6,6)
	results = {}
	ns = range(2,13) # number of states
	fntSize = 12
	plt.rcParams.update({'font.size': fntSize})


	for n in ns:
		with open(f"{filename}_{n}_{seed}.json") as f:
			results[n] = json.load(f)
	ylabels = ['log-likelihood', 'KS-test score']
	scales = [1, 1]
	for i, idx in enumerate(['likelihood', 'K-S', 'KLD']):		
		fig = plt.figure(figsize=fgsz)
		ax1 = fig.add_subplot(111)
		training = [results[n][idx + f'_{seed}'][0] for n in ns]
		test = [results[n][idx + f'_{seed}'][1]  for n in ns]
		ax1.plot(ns, training, '-.')
		ax1.plot(ns, test, ':')
		ax1.legend(['Training', 'Test'])
		ax1.set_xlabel('n')
		ax1.set_ylabel(ylabels[i])
		fmmts = matplotlib.ticker.ScalarFormatter()
		fmmts.set_powerlimits((-4,3))
		ax1.yaxis.set_major_formatter(fmmts)
		auxFs.clean_axes(ax1)
		fig.savefig(f"{filename}_{idx}.pdf")


if __name__ == '__main__':

	flname = sys.argv[1]
	seed = int(sys.argv[2])

	main(flname, seed)
