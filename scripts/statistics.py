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
def plot_freq(data):
    freqs = {i:0 for i in range(-2, 50)}
    for _, changes in data.items():
        deleted = len(changes["delete"])
        added = len(changes["create"])
        updated = len(changes["update"])
        total = updated + added - deleted
        freqs[total] += 1
    ax = plt.gca()
    y = range(1, sorted([i for i in freqs.keys() if freqs[i] > 0])[-1])
    bars = ax.bar(y, [freqs[i] for i in y])
    # for bar in bars:
    #     yval = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width() / 2, yval + 130, yval, ha="center")
    #ax.set_ylim(0, max(freqs.values()) * 1.18)
    steps = range(0, max(y) + 1, 5)
    plt.xticks(steps, map(str, steps))

@saver
def plot_overview(data):
    updated = 0
    added = 0
    deleted = 0
    labels = ["updated", "created", "deleted"]
    for _, changes in data.items():
        deleted += len(changes["delete"])
        added += len(changes["create"])
        updated += len(changes["update"])
    ax = plt.gca()
    y = range(len(labels))
    bars = ax.bar(y, [updated, added, deleted], color="darkslategray")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 130, yval, ha="center")
    ax.set_ylim(0, max([updated, added, deleted]) * 1.18)
    plt.xticks(y, labels)

@saver
def plot_type(data):
    types = ["car", "pedestrian", "bicycle", "train", "unknown"]
    cnt = {t: 0 for t in types}
    for _, changes in data.items():
        for _, c in changes["create"].items():
            if "type" in c["attributes"].keys():
                cnt[c["attributes"]["type"]] += 1
        for _, attr in changes["update"].items():
            if "type" in attr.keys():
                cnt[attr["type"]] += 1
    ax = plt.gca()
    y = range(len(types))
    bars = ax.bar(y, [cnt[t] for t in types])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 130, yval, ha="center")
    ax.set_ylim(0, max(cnt.values()) * 1.18)
    plt.xticks(y, types, rotation=30, ha="right")

@saver
def plot_size(data):
    types = ["car", "pedestrian", "bicycle", "train", "unknown"]
    cnt = {t: 0 for t in types}
    for _, changes in data.items():
        for _, c in changes["create"].items():
            if "type" in c["attributes"].keys():
                cnt[c["attributes"]["type"]] += 1
        for _, attr in changes["update"].items():
            if "type" in attr.keys():
                cnt[attr["type"]] += 1
    ax = plt.gca()
    y = range(len(types))
    bars = ax.bar(y, [cnt[t] for t in types])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 130, yval, ha="center")
    ax.set_ylim(0, max(cnt.values()) * 1.18)
    plt.xticks(y, types, rotation=30, ha="right")

@saver
def plot_relevance(data):
    types = ["ego relevant", "car visible irrelevant", "bicycle visible", "pedestrian visible", "other"]
    cnt = {t: 0 for t in types}
    def get_cat(attrs):
        if attrs["relevant"] == "yes":
            cnt["ego relevant"] += 1
        elif attrs["relevant"] == "no" and attrs["type"] == "car" and attrs["visible"] == "yes":
            cnt["car visible irrelevant"] += 1
        elif attrs["visible"] == "yes":
            if attrs["type"] == "pedestrian":
                cnt["pedestrian visible"] += 1
            if attrs["type"] == "bicycle":
                cnt["bicycle visible"] += 1
        else:
            cnt["other"] += 1
    for _, changes in data.items():
        for _, c in changes["create"].items():
            if "type" in c["attributes"].keys() and "visible" in c["attributes"].keys() and "relevant" in c["attributes"].keys():
                get_cat(c["attributes"])
        for _, attr in changes["update"].items():
            if "type" in attr.keys() and "visible" in attr.keys() and "relevant" in attr.keys():
                get_cat(attr)
    ax = plt.gca()
    y = range(len(types))
    bars = ax.bar(y, [cnt[t] for t in types])
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 130, yval, ha="center")
    ax.set_ylim(0, max(cnt.values()) * 1.3)
    plt.xticks(y, types, rotation=30, ha="right")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("changeset")
    args = parser.parse_args()

    data = read_json(args.changeset)
    ensure_dir(ODIR)
    plot_freq(data, sz=(2.3, 1.5), outfile="freq.pdf")
    plot_overview(data, sz=(2.3, 1.5), outfile="overview.pdf")
    plot_type(data, sz=(2.3, 1.5), outfile="types.pdf")
    plot_relevance(data, sz=(2.3, 1.5), outfile="relevance.pdf")
