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
FILE_PATTERN = "gtFine/*/*/*gtFine_polygons.json"

def get_by_label(labels, label):
    return [obj for obj in labels["objects"] if obj["label"] == label and not ("deleted" in obj.keys() and obj["deleted"])]

def check_keys(comp):
    for c in comp:
        if c not in KEYS:
            return False
    for k in KEYS:
        if k not in comp:
            return False
    return True

def check_values(key, choices, actual):
    if not actual:
        return False
    if actual not in choices:
        return False
    return True

def check_attr_map(attrs):
    comp = sorted(attrs.keys())
    if not check_keys(comp):
        return False
    for k in KEYS:
        if k in comp:
            if not check_values(k, SCHEMA[k], attrs[k]):
                return False
    return True

def check_attributes(obj):
    if not "attributes" in obj.keys():
        return False
    return check_attr_map(obj["attributes"])

def check_has_id(fn):
    with open(fn, "r") as f:
        labels = json.load(f)
        for obj in get_by_label(labels, LABEL):
            if not "id" in obj.keys():
                print("No ID found in {}".format(fn))

def dump_file_contents(fn, writer: csv.DictWriter):
    with open(fn, "r") as f:
        labels = json.load(f)
        for obj in get_by_label(labels, LABEL):
            if check_attributes(obj):
                attrs = obj["attributes"]
                keys = [Dump.RELEVANT.value, Dump.STATE.value, Dump.TYPE.value, Dump.VISIBLE.value]
                data = {Dump.PATH.value: fn,
                        Dump.ID.value: obj["id"]}
                data.update({k: attrs[k] for k in keys})
                writer.writerow(data)

def read_json(fn):
    with open(fn, "r") as f:
        data = json.load(f)
        return data

def get_idxs(obj,label="traffic light"):
    return [idx for idx, elem in enumerate(obj) if elem["label"] == label]

def compare_obj(orig, new):
    return orig["polygon"] == new["polygon"]

def compare_objs(orig, new):
    orig_idxs, new_idxs = get_idxs(orig), get_idxs(new)
    new_to_orig = {n: o for n, o in itertools.product(new_idxs, orig_idxs) if compare_obj(orig[o], new[n])}
    to_create = [n for n in new_idxs if n not in new_to_orig.keys()]
    to_delete = [o for o in orig_idxs if o not in new_to_orig.values()]
    return new_to_orig, to_delete, to_create

def make_changeset(new, changes):
    new_to_orig, to_delete, to_create = changes
    updates = {o_idx: new[n_idx]["attributes"] for n_idx, o_idx in new_to_orig.items()}
    create = {n_idx: new[n_idx] for n_idx in to_create}
    res = {"update": updates, "delete": to_delete, "create": create}
    return res

def changes_empty(changeset):
    return len(changeset["update"]) == 0 and len(changeset["delete"]) == 0 and len(changeset["create"]) == 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig-dir", required=True)
    parser.add_argument("--new-dir", required=True)
    parser.add_argument("--outfile", "-o", type=str, required=True)
    args = parser.parse_args()

    old_wd = os.getcwd()
    os.chdir(args.new_dir)
    files = glob.glob(FILE_PATTERN)
    if not files:
        raise RuntimeError("No files in " + args.basedir + " found")
    changes = {}
    for new_f in progressbar(files):
        old_f = os.path.join(args.orig_dir, new_f)
        orig_objs = read_json(old_f)["objects"]
        new_objs = read_json(new_f)["objects"]
        changeset = make_changeset(new_objs, compare_objs(orig_objs, new_objs))
        if not changes_empty(changeset) and "dusseldorf" in new_f:
            changes[new_f] = changeset
    with open(args.outfile, 'w') as of:
        json.dump(changes, of, indent=4, sort_keys=True)
    os.chdir(old_wd)
    print("Wrote changeset to {}".format(args.outfile))