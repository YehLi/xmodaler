import os
import numpy as np
import torch
import tqdm
import pickle

def read_lines(filepath):
    with open(filepath, "r") as f:
        return [e.strip("\n") for e in f.readlines()]

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/coco_sents.txt') as fid:
    sents = [line.strip() for line in fid]

vocab = read_lines('data/new_vocab_words.txt')
obj_dict = {w:i for i,w in enumerate(vocab)}

res = {}
data = load_pickle('data/coco_image2text_retrieval.pkl')
for image_id in data:
    wlist = []
    top_K = 7
    sents_ids = data[image_id][0][:top_K]
    sents_scores = data[image_id][1][:top_K]
    for sentid, score in zip(sents_ids, sents_scores):
        if score > 0.32:
            sent = sents[sentid]
            words = sent.split(' ')
            objs = [obj_dict[w] for w in words if w in obj_dict]
            wlist += objs
    wlist = list(set(wlist))
    res[image_id] = wlist

save_pickle(res, 'data/coco_image2text_retrieval_objs.pkl')
