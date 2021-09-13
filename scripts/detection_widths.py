#!/usr/bin/env python3

#!/usr/bin/env python3

import json
from enum import Enum, auto
import glob
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import os
import functools
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "9"
ODIR = "/tmp/plots"
LIM = 65

dist_reg= "/home/janosovits/width_reg.npy"
dist_large = "/home/janosovits/width_wide.npy"
dist_data = "/home/janosovits/width_data.npy"

def read_json(fn):
    with open(fn, "r") as f:
        data = json.load(f)
        return data

def ensure_dir(d):
    if os.path.exists(d) and not os.path.isdir(d):
        raise FileExistsError("{} exists and is not a directory".format(d))
    Path(d).mkdir(parents=True, exist_ok=True)

def saver(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        outfile = kwargs.pop("outfile") if "outfile" in kwargs.keys() else None
        sz = kwargs.pop("sz") if "sz" in kwargs.keys() else None
        plt.clf()
        if sz:
            plt.gcf().set_size_inches(sz[0], sz[1])
        value = func(*args, **kwargs)
        plt.tight_layout()
        if outfile:
            plt.savefig(os.path.join(ODIR, outfile), bbox_inches='tight', pad_inches=0)
        else:
            plt.show()
        return value
    return wrapper_decorator

@saver
def plot_reg():
    freqs = {i:0 for i in range(-2, 50)}
    data = np.load(dist_data)[:LIM]
    reg = np.load(dist_reg)
    shifted = [0] * 2000
    for i, r in enumerate(reg):
        shifted[2*i] = r
    shifted_s = shifted[:LIM]
    y = range(LIM)

    data_cum = np.cumsum(data).astype(np.float32)
    data_cum /= float(np.sum(data))
    shifted_cum = np.cumsum(shifted_s).astype(np.float32)
    shifted_cum /= float(np.sum(shifted))
    ax = plt.gca()
    ax.plot(y, data_cum)
    ax.plot(y, shifted_cum)
    # for bar in bars:
    #     yval = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width() / 2, yval + 130, yval, ha="center")
    #ax.set_ylim(0, max(freqs.values()) * 1.18)
    steps = range(0, max(y) + 1, 10)
    plt.xticks(steps, map(str, steps))

@saver
def plot_wide():
    data = np.load(dist_data)[:LIM]
    wide = np.load(dist_large)
    shifted_s = wide[:LIM]
    y = range(LIM)
    x2 = range(2000)
    data_cum = np.cumsum(data).astype(np.float32)
    data_cum /= float(np.sum(data))
    shifted_cum = np.cumsum(wide).astype(np.float32)
    shifted_cum /= float(np.sum(wide))
    ax = plt.gca()
    ax.plot(y, data_cum)
    ax.plot(x2, shifted_cum)
    # for bar in bars:
    #     yval = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width() / 2, yval + 130, yval, ha="center")
    #ax.set_ylim(0, max(freqs.values()) * 1.18)
    steps = range(0, max(y) + 1, 10)
    plt.xticks(steps, map(str, steps))

if __name__ == "__main__":
    ensure_dir(ODIR)
    plot_reg(sz=(2.3, 1.5), outfile="reg.pdf")
    plot_wide(sz=(2.3, 1.5), outfile="wide.pdf")
