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
CLASSES = ["egorel", "eogirel", "pedestr", "bike", "other"]

counts = {l: 0 for l in CLASSES}

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

def get_by_label(labels, label):
    return [obj for obj in labels["objects"] if obj["label"] == label and not ("deleted" in obj.keys() and obj["deleted"])]

def ensure_dir(d):
    if os.path.exists(d) and not os.path.isdir(d):
        raise FileExistsError("{} exists and is not a directory".format(d))
    Path(d).mkdir(parents=True, exist_ok=True)

def get_rect(obj):
    x,y = list(zip(*obj["polygon"]))
    return [int(min(x)), int(min(y)), int(max(x)), int(max(y))]

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("outdir", type=str)
    args = parser.parse_args()

    for c in CLASSES:
        ensure_dir(os.path.join(args.outdir, c))
    os.chdir(os.path.join(args.data, "gtFine"))
    files = glob.glob(FILE_PATTERN, recursive=True)
    print(os.getcwd())
    if not files:
        raise RuntimeError("No files in " + os.getcwd() + " found")
    changes = {}
    for new_f in progressbar(files):
        j = read_json(new_f)
        dump_if_nice(args.data, new_f, j, args.outdir)