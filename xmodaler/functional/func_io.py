import os
import numpy as np

def read_lines(path):
    with open(path, 'r') as fid:
        lines = [line.strip() for line in fid]
    return lines
    
def read_lines_set(path):
    lines = read_lines(path)
    lines = set(lines)
    return lines

# "features", "cls_prob", "boxes", "image_h", "image_w"
def read_np(image_path):
    content = np.load(image_path)
    keys = content.keys()
    if len(keys) == 1:
        return { "features": content[list(keys)[0]] }
    return content

def load_vocab(path):
    if len(path) == 0:
        return None
    vocab = ['.']
    with open(path, 'r') as fid:
        for line in fid:
            vocab.append(line.strip())
    return vocab