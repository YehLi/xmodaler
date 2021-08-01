# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import os
import copy
import json
import pickle
from tqdm import tqdm
import random
import numpy as np

from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.tokenization import BertTokenizer
from xmodaler.functional import (
    read_lines_set, 
    read_np, 
    boxes_to_locfeats, 
    dict_as_tensor
)

from ..build import DATASETS_REGISTRY

__all__ = ["MSCoCoBertDataset"]

def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)

def encode_sequences_bert(entries, tokenizer, cls_token_id, sep_token_id, pad_token_id, max_length):
    for entry in tqdm(entries, desc='BertTokenize Coco Seqs'):
        entry['tokens_ids'] = []
        entry['target_ids'] = []

        for sent in entry['caption']:
            tokens = tokenizer.encode(sent)
            input_seq = [cls_token_id] + tokens
            target_seq = tokens + [sep_token_id]

            input_seq = input_seq[: max_length]
            target_seq = target_seq[: max_length]

            if len(input_seq) < max_length:
                padding = [pad_token_id] * (max_length - len(input_seq))
                tpadding = [-1] * (max_length - len(input_seq))
                input_seq = input_seq + padding
                target_seq = target_seq + tpadding

            assert_eq(len(input_seq), max_length)
            assert_eq(len(target_seq), max_length)

            entry['tokens_ids'].append(input_seq)
            entry['target_ids'].append(target_seq)

        entry.pop('caption')
        entry['tokens_ids'] = np.array(entry['tokens_ids'], dtype='uint32')
        entry['target_ids'] = np.array(entry['target_ids'], dtype='int32')

    return entries


@DATASETS_REGISTRY.register()
class MSCoCoBertDataset:
    @configurable
    def __init__(
        self,
        stage: str,
        anno_file: str,
        seq_per_img: int,
        max_seq_length: int,
        max_feat_num: int,
        feats_folder: str,
        images_ids_file: str,
        tokenizer
    ):
        self.stage = stage
        self.anno_file = anno_file
        self.seq_per_img = seq_per_img
        self.max_seq_length = max_seq_length
        self.max_feat_num = max_feat_num
        self.feats_folder = feats_folder
        self.images_ids_file = images_ids_file
        self.tokenizer = tokenizer
        
    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        ann_files = {
            "train": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "dataset_coco.json"),
            "val": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "dataset_coco.json"),
            "test": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "dataset_coco.json"),
        }
        images_ids_files = {
            "train": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "coco_train_image_id.txt"),
            "val": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "coco_val_image_id.txt"),
            "test": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "coco_test_image_id.txt")
        }
        ret = {
            "stage": stage,
            "anno_file": ann_files[stage],
            "max_seq_length": cfg.MODEL.MAX_SEQ_LEN,
            "seq_per_img": cfg.DATALOADER.SEQ_PER_SAMPLE,
            "max_feat_num": cfg.DATALOADER.MAX_FEAT_NUM,
            "feats_folder": cfg.DATALOADER.FEATS_FOLDER,
            "images_ids_file": images_ids_files[stage],
            "tokenizer": BertTokenizer.from_pretrained(cfg.MODEL.PRETRAINING.MODEL_NAME,
                do_lower_case=cfg.MODEL.PRETRAINING.DO_LOWER_CASE),
        }
        return ret

    def load_data(self, cfg):
        # Load mscoco data like pretraining
        cache_files = {
            "train": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "mscoco_bert_caption_anno_train.pkl"),
            "val": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "mscoco_bert_caption_anno_val.pkl"),
            "test": os.path.join(cfg.DATALOADER.ANNO_FOLDER, "mscoco_bert_caption_anno_test.pkl")
        }
        if not os.path.exists(cache_files[self.stage]):
            images_ids_set = read_lines_set(self.images_ids_file)
            entries = []
            annotation = json.load(open(self.anno_file, 'r'))['images']
            for img in annotation:
                image_id = str(img['cocoid'])
                if image_id not in images_ids_set:
                    continue

                sentences = []
                for sent in img['sentences']:
                    sentences.append(sent['raw'].lower().strip().strip('.'))
                entries.append({"caption": sentences, "image_id": image_id})

            # pre-tokenize mscoco caption
            cls_token_id = self.tokenizer.vocab["[CLS]"]
            sep_token_id = self.tokenizer.vocab["[SEP]"]
            pad_token_id = self.tokenizer.vocab["[PAD]"]
            datalist = encode_sequences_bert(entries, self.tokenizer, cls_token_id, sep_token_id, pad_token_id, self.max_seq_length)
            pickle.dump(datalist, open(cache_files[self.stage], "wb"))
        else:
            datalist = pickle.load(open(cache_files[self.stage], "rb"))

        return datalist
        
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image_id = dataset_dict['image_id']    
        
        # load image feature
        image_path = os.path.join(self.feats_folder, image_id + ".npz")
        content = read_np(image_path)
        features = content['features'][0:self.max_feat_num - 1]
        boxes = content['boxes'][0:self.max_feat_num - 1]
        image_h = content['image_h'][0]
        image_w = content['image_w'][0]
        num_boxes = len(boxes)
        image_locations = boxes_to_locfeats(boxes, image_w, image_h)

        # add g_feat
        imgfeats = features
        g_image_feat = np.sum(imgfeats, axis=0) / num_boxes
        g_image_location = np.array([0, 0, 1, 1, 1])

        num_boxes = num_boxes + 1
        imgfeats = np.concatenate([np.expand_dims(g_image_feat, axis=0), imgfeats], axis=0)
        image_locations = np.concatenate([np.expand_dims(g_image_location, axis=0), image_locations], axis=0)

        # build visual output
        ret = {
            kfg.IDS: image_id,
            kfg.ATT_FEATS: imgfeats.astype('float32'),
            kfg.ATT_FEATS_LOC: image_locations.astype('float32')
        }

        # if test
        if self.stage != 'train':
            g_tokens_type = np.ones((self.max_seq_length,), dtype=np.int64)
            ret.update({ kfg.G_TOKENS_TYPE: g_tokens_type })
            dict_as_tensor(ret)
            return ret

        # load caption data
        sent_num = len(dataset_dict['tokens_ids'])
        if sent_num >= self.seq_per_img:
            selects = random.sample(range(sent_num), self.seq_per_img)
        else:
            selects = random.choices(range(sent_num), k = (self.seq_per_img - sent_num))
            selects += list(range(sent_num))

        tokens_ids = [ dataset_dict['tokens_ids'][i,:].astype(np.int64) for i in selects ]
        target_ids = [ dataset_dict['target_ids'][i,:].astype(np.int64) for i in selects ]
        g_tokens_type = [ np.ones((len(dataset_dict['tokens_ids'][i,:]), ), dtype=np.int64) for i in selects ]

        ret.update({
            kfg.SEQ_PER_SAMPLE: self.seq_per_img,
            kfg.G_TOKENS_IDS: tokens_ids,
            kfg.G_TARGET_IDS: target_ids,
            kfg.G_TOKENS_TYPE: g_tokens_type,
        })
        dict_as_tensor(ret)
        return ret
