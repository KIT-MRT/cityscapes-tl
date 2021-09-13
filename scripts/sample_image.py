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

class Dump(Enum):
    PATH = "relative_file_path"
    ID = "id"
    RELEVANT = "relevant"
    STATE = "state"
    TYPE = "type"
    VISIBLE = "visible"

SCHEMA = {"relevant": ["yes", "no"], "state": ["red", "yellow", "green", "off", "unknown"], "type": ["car", "pedestrian", "bicycle", "unknown", "train"], "visible": ["yes", "no"]}
KEYS = sorted(SCHEMA.keys())
LABEL = "traffic light"
FILE_PATTERN = "*/*/*gtFine_polygons.json"

def read_json(fn):
    with open(fn, "r") as f:
        data = json.load(f)
        return data

def get_color(attr):
    if attr["type"] == "car" and attr["relevant"] == "yes":
        return (255, 0, 255)
    elif attr["type"] == "car":
        return (255, 255, 0)
    # elif attr["type"] == "car" and attr["visible"] == "no":
    #     return (212, 187, 150)
    elif attr["type"] == "pedestrian":
        return (0, 255, 0)
    elif attr["type"] == "train":
        return (0, 240, 240)
    else:
        return (0, 0, 0)

def get_color_orig(attr):
    return (255, 255, 255)

def get_by_label(labels, label):
    return [obj for obj in labels["objects"] if obj["label"] == label and not ("deleted" in obj.keys() and obj["deleted"])]

def ensure_dir(d):
    if os.path.exists(d) and not os.path.isdir(d):
        raise FileExistsError("{} exists and is not a directory".format(d))
    Path(d).mkdir(parents=True, exist_ok=True)

def get_rect(obj):
    x,y = list(zip(*obj["polygon"]))
    return (int(min(x)), int(min(y))), (int(max(x)), int(max(y)))

def draw_lights(img_path, objs, outdir, color_func, bw):
    if not os.path.exists(img_path):
        raise FileNotFoundError("{} does not exist".format(img_path))
    arr = cv2.imread(img_path)
    if bw:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    for tl in objs:
        p1, p2 = get_rect(tl)
        cv2.rectangle(arr, p1, p2, color_func(tl["attributes"]), thickness=2)
    out_path = os.path.join(outdir, os.path.basename(img_path))
    if bw:
        out_path = out_path.replace(".png", "_bw.png")
    else:
        out_path = out_path.replace(".png", "_color.png")
    cv2.imwrite(out_path, arr)

def dump_if_nice(basedir, f, objects, outdir):
    tls = get_by_label(objects, LABEL)
    if len(tls) > 17:
        imgp = os.path.join(basedir, "leftImg8bit", f)
        imgp = imgp.replace("gtFine_polygons.json", "leftImg8bit.png")
        draw_lights(imgp, tls, outdir, get_color, False)
        draw_lights(imgp, tls, outdir, get_color_orig, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("outdir", type=str)
    args = parser.parse_args()

    ensure_dir(args.outdir)
    old_wd = os.getcwd()
    os.chdir(os.path.join(args.data, "gtFine"))
    files = glob.glob(FILE_PATTERN)
    if not files:
        raise RuntimeError("No files in " + os.getcwd() + " found")
    changes = {}
    for new_f in progressbar(files):
        j = read_json(new_f)
        dump_if_nice(args.data, new_f, j, args.outdir)