#!/usr/bin/env python3

import json
from enum import Enum, auto
import glob
import argparse
from progressbar import progressbar
import sys
import os
import csv
import itertools
from PIL import Image
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import functools
import numpy as np
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

LABEL = "traffic light"
FILE_PATTERN = "./*/*/*gtFine_polygons.json"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "9"
ODIR = "/tmp/plots"
SZ = 1024, 2048
DIS = 8

xg, yg = np.meshgrid(np.linspace(0, SZ[1] / DIS, int(SZ[1] / DIS + 1)), np.linspace(0, SZ[0], int(SZ[0] / DIS + 1)))
zg = np.zeros_like(xg)
print(zg.shape)

def ensure_dir(d):
    if os.path.exists(d) and not os.path.isdir(d):
        raise FileExistsError("{} exists and is not a directory".format(d))
    Path(d).mkdir(parents=True, exist_ok=True)

LIM = 65

counts = {l: 0 for l in range(LIM + 2)}

def read_json(fn):
    with open(fn, "r") as f:
        data = json.load(f)
        return data

def get_label(attr):
    if "type" in attr.keys() and "relevant" in attr.keys():
        if attr["type"] == "car" and attr["relevant"] == "yes":
            return "egorel"
        elif attr["type"]== "car" and attr["relevant"] == "no":
            return "eogirel"
        elif attr["type"] == "pedestrian":
            return "pedestr"
        elif attr["type"] == "bicycle":
            return "bike"
        else:
            return "other"
    else:
        return None

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

def get_by_label(labels, label):
    return [obj for obj in labels["objects"] if obj["label"] == label and not ("deleted" in obj.keys() and obj["deleted"])]

def ensure_dir(d):
    if os.path.exists(d) and not os.path.isdir(d):
        raise FileExistsError("{} exists and is not a directory".format(d))
    Path(d).mkdir(parents=True, exist_ok=True)

def get_rect(obj):
    x,y = list(zip(*obj["polygon"]))
    return int(max(x)) - int(min(x))

def get_corner(obj):
    x,y = list(zip(*obj["polygon"]))
    return int(min(x)), int(min(y))

def write_crop(img, tl, cls, outdir):
    rect = get_rect(tl)
    # print(rect)
    # print(img.shape)
    crop_img = img[rect[1]:rect[3], rect[0]:rect[2]]
    out_path = os.path.join(outdir, cls, "{:06d}.png".format(counts[cls]))
    counts[cls] += 1
    if crop_img.size > 0:
        cv2.imwrite(out_path, crop_img)

def dump_if_nice(basedir, f, objects, outdir):
    tls = get_by_label(objects, LABEL)
    if len(tls) > 0:
        imgp = os.path.join(basedir, "leftImg8bit", f)
        imgp = imgp.replace("gtFine_polygons.json", "leftImg8bit.png")
        if not os.path.exists(imgp):
            raise FileNotFoundError("{} does not exist".format(imgp))
        arr = cv2.imread(imgp)
        for tl in tls:
            cls = get_label(tl["attributes"])
            if cls is not None:
                write_crop(arr, tl, cls, outdir)

def add_to_statistic(objects):
    tls = get_by_label(objects, LABEL)
    if len(tls) > 0:
        for tl in tls:
            sz = get_rect(tl)
            if sz <= LIM:
                counts[sz] += 1
            else:
                counts[LIM + 1] += 1
            corner = get_corner(tl)
            zg[int(corner[1] / DIS), int(corner[0] / DIS)] += 1

@saver
def make_plot():
    ax = plt.gca()
    y = range(1, LIM + 2)
    bars = ax.bar(y, [counts[i] for i in y])
    # for bar in bars:
    #     yval = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width() / 2, yval + 130, yval, ha="center")
    #ax.set_ylim(0, max(freqs.values()) * 1.18)
    steps = list(range(5, max(y) - 2, 10))
    ticks = list(map(str, steps))
    steps.append(66)
    ticks.append(">65")
    plt.xticks(steps, ticks)
    vals = [counts[i] for i in y]
    arr = np.array(vals)
    np.save("/tmp/width_data.npy", arr)

@saver
def make_heatmap(zg):
    ax = plt.gca()
    zg = np.flipud(zg)
    z_max = zg.max()
    #c = ax.pcolormesh(xg, yg, zg, cmap='Greys', vmin=0, vmax=z_max, norm=colors.LogNorm(vmin=0, vmax=z_max))
    c= ax.pcolormesh(xg, yg, zg, cmap='Greys', norm=colors.LogNorm(vmin=0.1, vmax=z_max))
    # set the limits of the plot to the limits of the data

    axins = inset_axes(ax,
                       width="3%",  # width = 5% of parent_bbox width
                       height="100%",  # height : 50%
                       loc='lower left',
                       bbox_to_anchor=(-0.05, 0., 1, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0,
                       )
    ax.axis([xg.min(), xg.max(), yg.min(), yg.max()])
    ax.set_xticks([])
    ax.set_yticks([])
    bar = plt.gcf().colorbar(c, cax=axins)
    axins.yaxis.set_ticks_position('left')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    #parser.add_argument("outdir", type=str)
    args = parser.parse_args()

    os.chdir(os.path.join(args.data, "gtFine"))
    files = glob.glob(FILE_PATTERN, recursive=True)
    print(os.getcwd())
    if not files:
        raise RuntimeError("No files in " + os.getcwd() + " found")
    changes = {}
    for new_f in progressbar(files):
        j = read_json(new_f)
        add_to_statistic(j)
    print(counts)
    ensure_dir(ODIR)
    make_plot(sz=(2.3, 1.5), outfile="size.pdf")
    make_heatmap(zg, sz=(5, 2.2), outfile="scatter.pdf")