#!/usr/bin/env python3

import json
import argparse
import os
from progressbar import progressbar
from pathlib import Path
import glob

LABEL = "traffic light"
FILE_PATTERN = "gtFine/*/*/*gtFine_polygons.json"

def my_marginalization(object):
    if not "relevant" in object["attributes"].keys():
        return
    if object["attributes"]["relevant"] == "yes":
        object["label"] = "tl relevant"
    else:
        object["label"] = "tl irrelevant"
    return object

def read_json(fn):
    with open(fn, "r") as f:
        data = json.load(f)
        return data

def ensure_dir(d):
    if os.path.exists(d) and not os.path.isdir(d):
        raise FileExistsError("{} exists and is not a directory".format(d))
    Path(d).mkdir(parents=True, exist_ok=True)

def update_objects(objects, func, label=LABEL):
    for i, obj in enumerate(objects):
        if obj["label"] == label:
            objects[i] = func(obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("basedir")
    parser.add_argument("target_dir")
    args = parser.parse_args()

    old_wd = os.getcwd()
    os.chdir(args.basedir)
    files = glob.glob(FILE_PATTERN)
    if not files:
        raise RuntimeError("No files in " + args.basedir + " found")
    for old_f in progressbar(files):
        data = read_json(old_f)
        update_objects(data["objects"], my_marginalization)
        target_file = os.path.join(args.target_dir, old_f)
        ensure_dir(os.path.dirname(target_file))
        with open(target_file, "w") as t:
            json.dump(data, t, indent=4, sort_keys=True)

