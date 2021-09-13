#!/usr/bin/env python3

import json
from enum import Enum, auto
import glob
import argparse
from progressbar import progressbar

class LabelError(Enum):
    NO_ATTRIBUTES = auto()
    MISSING_TAG = auto()
    INVALID_TAG = auto()
    MISSING_VALUE = auto()
    INVALID_VALUE = auto()

SCHEMA = {"relevant": ["yes", "no"], "state": ["red", "yellow", "green", "off", "unknown"], "type": ["car", "pedestrian", "bicycle", "unknown", "train"], "visible": ["yes", "no"]}
KEYS = sorted(SCHEMA.keys())
LABEL = "traffic light"
FILE_PATTERN = "/gtFine/*/*/*gtFine_polygons.json"

def get_by_label(labels, label):
    return [obj for obj in labels["objects"] if obj["label"] == label and not ("deleted" in obj.keys() and obj["deleted"])]

def check_keys(comp):
    errs = []
    for c in comp:
        if c not in KEYS:
            errs.append((LabelError.INVALID_TAG, c))
    for k in KEYS:
        if k not in comp:
            errs.append((LabelError.MISSING_TAG, k))
    return errs

def check_values(key, choices, actual):
    if not actual:
        return [(LabelError.MISSING_VALUE, (key, actual))]
    if actual not in choices:
        return [(LabelError.INVALID_VALUE, (key, actual))]
    return []

def check_attr_map(attrs):
    errs = []
    comp = sorted(attrs.keys())
    errs += check_keys(comp)
    for k in KEYS:
        if k in comp:
            errs += check_values(k, SCHEMA[k], attrs[k])
    return errs

def check_attributes(obj):
    errs = []
    if not "attributes" in obj.keys():
        errs.append([(LabelError.NO_ATTRIBUTES)])
        return errs
    errs += check_attr_map(obj["attributes"])
    return errs

def check_file(fn):
    with open(fn, "r") as f:
        labels = json.load(f)
        return [(int(obj["id"]) if "id" in obj.keys() else None,  check_attributes(obj)) for obj in get_by_label(labels, LABEL) if check_attributes(obj) != []]

def to_string(err):
    if err[0] == LabelError.NO_ATTRIBUTES:
        return "no attributes found"
    if err[0] == LabelError.MISSING_TAG:
        return "Tag \"{}\" not found".format(err[1])
    if err[0] == LabelError.MISSING_VALUE:
        return "Value \"{}\" not found".format(err[1])
    if err[0] == LabelError.INVALID_TAG:
        return "Tag \"{}\"is not valid".format(err[1])
    if err[0] == LabelError.INVALID_VALUE:
        return "Value \"{}\" for key \"{}\" is not valid".format(err[1][1], err[1][0])
    else:
        raise RuntimeError("invalid error code")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("basedir")
    parser.add_argument("--outfile", "-o", type=str)
    args = parser.parse_args()

    files = glob.glob(args.basedir + FILE_PATTERN)
    if not files:
        raise RuntimeError("No files in " + args.basedir + " found")
    msg = []
    for f in progressbar(files):
        errmap = check_file(f)
        for lid, errlist in errmap:
            if errlist:
                print(errmap)
                for err in errlist:
                    msg.append("File {} item ID {}: {}\n".format(f, lid, to_string(err)))
    if args.outfile:
        with open(args.outfile, "w") as f:
            f.writelines(msg)
    else:
        for l in msg:
            print(l)
