#!/usr/bin/env python3

import json
import argparse
import os
from progressbar import progressbar

def delete_from_list(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)

def update(objects, idx, attrs):
    objects[int(idx)]["attributes"] = attrs

def apply_change_to_objects(objects: list, changes: dict):
    for idx, change in changes["update"].items():
        update(objects, idx, change)
    delete_from_list(objects, changes["delete"])
    for idx, change in changes["create"].items():
        objects.insert(int(idx), change)

def apply_change_to_file(fn, change, dry_run):
    with open(fn, "r") as f:
        root = json.load(f)
        apply_change_to_objects(root["objects"], change)
    if not dry_run:
        with open(fn, "w") as f:
            json.dump(root, f,  indent=4, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("basedir")
    parser.add_argument("changeset")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.changeset) as cfile:
        changes = json.load(cfile)
        for fn, change in progressbar(changes.items()):
            apply_change_to_file(os.path.join(args.basedir, fn), change, args.dry_run)