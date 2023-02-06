  
import os
import json
import zipfile
import numpy as np
import pickle
import xmltodict

def load_xml(filename):
    with open(filename, 'r') as f:
        obj = xmltodict.parse(f.read())
    return json.loads(json.dumps(obj))

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def save_jsonl(data, filename):
    """data is a list"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e) for e in data]))


def save_lines(list_of_str, filepath):
    with open(filepath, "w") as f:
        f.write("\n".join(list_of_str))


def read_lines(filepath):
    with open(filepath, "r") as f:
        return [e.strip("\n") for e in f.readlines()]


def mkdirp(p):
    if not os.path.exists(p):
        os.makedirs(p)


def flat_list_of_lists(l):
    """flatten a list of lists [[1,2], [3,4]] to [1,2,3,4]"""
    return [item for sublist in l for item in sublist]
