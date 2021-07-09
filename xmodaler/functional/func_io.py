import os
import numpy as np
from .func_feats import boxes_to_locfeats

def read_lines(path):
    with open(path, 'r') as fid:
        lines = [line.strip() for line in fid]
    return lines
    
def read_lines_set(path):
    lines = read_lines(path)
    lines = set(lines)
    return lines

# "features", "cls_prob", "boxes", "image_h", "image_w"
def read_np(path):
    content = np.load(path)
    if isinstance(content, np.ndarray):
        return { "features": content } 

    keys = content.keys()
    if len(keys) == 1:
        return { "features": content[list(keys)[0]] }
    return content

def read_np_bbox(path, max_feat_num):
    content = read_np(path)
    features = content['features'][0:max_feat_num - 1]
    boxes = content['boxes'][0:max_feat_num - 1]
    image_h = content['image_h'][0]
    image_w = content['image_w'][0]
    num_boxes = len(boxes)

    g_feat = np.sum(features, axis=0) / num_boxes
    features = np.concatenate([np.expand_dims(g_feat, axis=0), features], axis=0)

    image_locations = boxes_to_locfeats(boxes, image_w, image_h)
    g_location = np.array([0, 0, 1, 1, 1])
    image_locations = np.concatenate([np.expand_dims(g_location, axis=0), image_locations], axis=0)
    return features, image_locations


def load_vocab(path):
    if len(path) == 0:
        return None
    vocab = ['.']
    with open(path, 'r') as fid:
        for line in fid:
            vocab.append(line.strip())
    return vocab