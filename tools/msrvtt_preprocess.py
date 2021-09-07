"""
Preprocess raw json dataset into pkl/json files for use in xmodaler/datasets/videos/msrvtt.py

Input: json files (both videodatainfo_2016.json and test_videodatainfo_2016.json) that have the form
{
  "info": {"contributor": "MSRA MSM Group", ...},
  "videos": [{"category": 9, "url": "https://xxx", "video_id": "video0", "split": "train", ...}, ...]
  "sentences": [{"caption": "xx xxx", "video_id": "video0", "sen_id": 0}, ...] 
}

This script reads these two json files, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: json and pkl files as well as vocabulary.txt

To run this script in bash and save the files in current work directory:
python3 msrvtt_preprocess.py --input_json videodatainfo_2016.json test_videodatainfo_2016.json --output_dir .
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
# import h5py
import numpy as np
import torch
import torchvision.models as models
# import skimage.io
# from PIL import Image
import pickle as pkl

def build_vocab(sentences, vid2split, params):
    count_thr = params['word_count_threshold']

    # count up the number of words
    counts = {}
    for sent in sentences:
        vid = sent['video_id'] # "video123"
        tokens = sent['caption'].split(' ')
        sent['tokens'] = tokens
        sent["split"] = vid2split[vid]
        if vid2split[vid] == "train":
            pass
        else:
            continue # skip val, test data
        for w in tokens:
            counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
    print('top words and their counts:')
    print('\n'.join(map(str,cw[:20])))

    # print some stats
    total_words = sum(counts.values())
    print('total words:', total_words)
    bad_words = [w for w,n in counts.items() if n <= count_thr]

    vocab = [w for w,n in counts.items() if n > count_thr]

    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
    print('number of words in vocab would be %d' % (len(vocab), ))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for sent in sentences:
        txt = sent['tokens']
        nw = len(txt)
        sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    print('max length sentence in raw data: ', max_len)
    print('sentence length distribution (count, number of words):')
    sum_len = sum(sent_lengths.values())
    for i in range(max_len+1):
        print('%2d: %10d   %f%%' % (i, sent_lengths.get(i,0), sent_lengths.get(i,0)*100.0/sum_len))

    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('UNK')

    for sent in sentences:
        txt = sent['tokens']
        sent['final_captions'] = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]

    return vocab

def encode_captions(sentences, params, wtoi):
    """ 
    encode captions of the same video into one 2-D array,
    also produces a dict to point out the array based on image id
    """
    max_length = params['max_length']

    datalist = {"train": [], "val": [], "test": []}
    visited = set()
    #min_cap_count = 10000
    for sent in sentences:
        split = sent["split"]
        img_id = sent["video_id"].replace("video", "") # 'video123'

        input_Li = np.zeros((1, max_length + 1), dtype='uint32')
        output_Li = np.zeros((1, max_length + 1), dtype='int32') - 1

        for k, w in enumerate(sent['final_captions']):
            if k < max_length:
                input_Li[0,k+1] = wtoi[w] # one shift for <BOS>
                output_Li[0,k] = wtoi[w]

        seq_len = len(sent['final_captions'])
        if seq_len <= max_length:
            output_Li[0, seq_len] = 0
        else:
            output_Li[0, max_length] = wtoi[sent['final_captions'][max_length]] 
        
        if split != 'train' and img_id in visited:
            continue
        visited.add(img_id)
        new_data = {
            "video_id": int(img_id),
            "tokens_ids": input_Li,
            "target_ids": output_Li,
        }
        datalist[split].append(new_data)
    
    print("number of train videos ", len(datalist["train"]))
    print("number of val videos ", len(datalist["val"]))
    print("number of test videos ", len(datalist["test"]))
    
    return datalist

def save_pkl_file(datalist, output_dir):
    for split in datalist:
        pkl.dump(datalist[split], open(os.path.join(output_dir, "msrvtt_caption_anno_{}.pkl".format(split)), "wb"))

def save_id_file(imgs, output_dir):
    ids = {"train": [], "val": [], "test": []}

    for img in imgs:
        split = img["split"]
        img_id = img["imgid"]
        if split == "train":
            for j, _ in enumerate(img["sentences"]):
                ids[split].append("{}_{}".format(img_id, j))
        else:
            ids[split].append(img_id)

    for split, _ids in ids.items():
        with open(os.path.join(output_dir, "{}_ids.txt".format(split)), "w") as fout:
            for imgid in _ids:
                fout.write("{}\n".format(imgid))

def save_split_json_file(sentences, output_dir):
    split_data = {  "train": {"images": [], "annotations": []}, 
                    "val": {"images": [], "annotations": []}, 
                    "test": {"images": [], "annotations": []}, 
                    }

    visited = set()
    for sent in sentences:
        split = sent["split"]

        if sent["video_id"] not in visited:
            new_image = {
                "id": int(sent["video_id"].replace("video","")),
                "file_name": sent["video_id"]
            }
            split_data[split]["images"].append(new_image)
            visited.add(sent["video_id"])
        else:
            pass

        new_caption = {
            "image_id": int(sent["video_id"].replace("video","")),
            "id": sent["sen_id"],
            "caption": sent["caption"]
        }
        split_data[split]["annotations"].append(new_caption)

    for split, data in split_data.items():
        if split == "train":
	        continue
        json.dump(data, open(os.path.join(output_dir, "captions_{}.json".format(split)), "w") )

def build_vid_split_mapping(videos):
    vid2split = {}
    for video in videos:
        vid = video['video_id'] # "video123"
        split = 'val' if video['split'] == 'validate' else video['split']
        vid2split[vid] = split
    return vid2split

def load_list(datafiles):
    videos = []     
    sentences = []
    for f in datafiles:
        data = json.load(open(f, 'r'))
        videos += data["videos"]
        sentences += data["sentences"]

    vid2split = build_vid_split_mapping(videos)
    return sentences, videos, vid2split

def main(params):
    if not os.path.exists(params["output_dir"]):
        os.makedirs(params["output_dir"])

    sentences, videos, vid2split = load_list(params['input_json'])
    print("Number of data: ", len(sentences))
    
    seed(123) # make reproducible
    
    # create the vocab
    vocab = build_vocab(sentences, vid2split, params)
    itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
    wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table

    print("Vocabulary size: ", len(vocab))
    with open(os.path.join(params["output_dir"], "vocabulary.txt"), "w") as fout:
        for w in vocab:
            fout.write("{}\n".format(w))
    
    # encode captions in large arrays, ready to ship to hdf5 file
    datalist = encode_captions(sentences, params, wtoi)

    # create output file
    save_pkl_file(datalist, params['output_dir'])
    save_split_json_file(sentences, params['output_dir'])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', default='dataset.json', type=str, nargs='+', help='input json file to process into hdf5')
    parser.add_argument('--output_dir', default='.', help='output directory')

    # options
    parser.add_argument('--max_length', default=20, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=5, type=int, help='only words that occur more than this number of times will be put in vocab')
    
    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent = 2))
    main(params)
