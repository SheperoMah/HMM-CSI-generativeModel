#!/usr/bin/env python3


import sys
import matplotlib.pyplot as plt
import auxiliaryFuncs as auxFs
import json

def read_json(fn):
    with open(fn, 'r') as f:
	       dictionary = json.load(f)
    return(dictionary)


def main():
    maeDict1 = read_json(f"Ahawaii_acf_mae.json")
    maeDict2 = read_json(f"A_acf_mae.json")
    labels = ['Hawaii', 'Norrköping']
    ns = range(2,13)
    fgsz = (6,6)
    seed = 100

    trainTimeHawaii, trainTimeNorr = {}, {}
    for n in ns:
        trainTimeHawaii[n] = read_json(f"Ahawaii_{n}_{seed}.json")
        trainTimeNorr[n] = read_json(f"A_{n}_{seed}.json")

    # Plot the acf error as function of the state
    fig1 = plt.figure(figsize=fgsz)
    ax1 = fig1.add_subplot(111)
    ax1.plot(ns, maeDict1.values())
    ax1.plot(ns, maeDict2.values())
    ax1.set_xlabel('n')
    ax1.set_ylabel('ACF MAE')
    ax1.legend(labels)
    auxFs.clean_axes(ax1)
    fig1.savefig(f"acf_mae.pdf")

    # Plot the training time
    fig2 = plt.figure(figsize=fgsz)
    ax2 = fig2.add_subplot(111)
    timeH = [trainTimeHawaii[n][f'train_time_{seed}'] for n in ns]
    timeN = [trainTimeNorr[n][f'train_time_{seed}'] for n in ns]
    ax2.plot(ns, timeH)
    ax2.plot(ns, timeN)
    ax2.set_xlabel('n')
    ax2.legend(labels)
    ax2.set_ylabel('Training time (second)')
    auxFs.clean_axes(ax2)
    fig2.savefig(f"train_time.pdf")


if __name__ == "__main__":

	main()
