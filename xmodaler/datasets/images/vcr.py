# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import os
import copy
import pickle
import random
import json
import json_lines
import numpy as np
from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import read_np, dict_as_tensor, boxes_to_locfeats, read_np_bbox
from xmodaler.tokenization import BertTokenizer
from ..build import DATASETS_REGISTRY

__all__ = ["VCRDataset"]

@DATASETS_REGISTRY.register()
class VCRDataset:
    @configurable
    def __init__(
        self,
        stage: str,
        task_name: str,  # VCR_Q-A or VCR_QA-R
        anno_folder: str,
        feats_folder: str,
        max_feat_num: int,
        max_seq_len: int,
        seq_per_img: int,
        tokenizer
    ):
        self.stage = stage
        self.task_name = task_name
        self.anno_folder = anno_folder
        self.feats_folder = feats_folder
        self.gt_feat_folder = feats_folder + "_gt"
        self.max_feat_num = max_feat_num
        self.seq_per_img = seq_per_img
        self.tokenizer = tokenizer

        if self.task_name == 'VCR_Q-A':
            self.max_seq_len = 38
        else:
            self.max_seq_len = 66

        self.names = [
            'Casey', 'Riley', 'Jessie', 'Jackie', 
            'Avery', 'Jaime', 'Peyton', 'Kerry', 
            'Jody', 'Kendall', 'Frankie', 'Pat', 'Quinn']

    
    @classmethod
    def from_config(cls, cfg, stage: str = "train;VCR_Q-A"):
        stage, task_name = stage.split(';')

        ret = {
            "stage": stage,
            "task_name": task_name,
            "anno_folder": cfg.DATALOADER.ANNO_FOLDER,
            "feats_folder": cfg.DATALOADER.FEATS_FOLDER,
            "max_feat_num": cfg.DATALOADER.MAX_FEAT_NUM,
            "max_seq_len": cfg.MODEL.MAX_SEQ_LEN,
            "seq_per_img": cfg.DATALOADER.SEQ_PER_SAMPLE,
            "tokenizer": BertTokenizer.from_pretrained(cfg.MODEL.PRETRAINING.MODEL_NAME,
                do_lower_case=cfg.MODEL.PRETRAINING.DO_LOWER_CASE),
        }
        return ret

    def load_data(self, cfg):
        cache_path = os.path.join(
            self.anno_folder, "cache",
            "%s_%s_%d.pkl" % (self.task_name, self.stage, self.max_seq_len)
        )
        if not os.path.exists(cache_path):
            datalist = self.load_raw_data(cfg)
            self.tokenize(datalist)
            pickle.dump(datalist, open(cache_path, "wb"))
        datalist = pickle.load(open(cache_path, "rb"))
        return datalist

    def tokenize(self, datalist):
        person_name_id = 0
        for entry in datalist:
            objects_replace_name = []
            for o in entry['objects']:
                if o == 'person':
                    objects_replace_name.append(self.names[person_name_id])
                    person_name_id = (person_name_id + 1) % len(self.names)
                else:
                    objects_replace_name.append(o)

            tokens_q = self.retokenize_and_convert_to_ids(entry["question"], objects_replace_name)
            if self.task_name == "VCR_QA-R":
                tokens_q2 = self.retokenize_and_convert_to_ids(entry["question_a"], objects_replace_name)

            tokens_qr_arr = []
            for answer in entry["answers"]:
                tokens_r = self.retokenize_and_convert_to_ids(answer, objects_replace_name)

                if self.task_name == "VCR_Q-A":
                    tokens_q_copy = copy.copy(tokens_q)
                    self.truncate_seq_pair(tokens_q_copy, tokens_r, self.max_seq_len - 3)
                else:
                    tokens_q_copy = copy.copy(tokens_q)
                    tokens_q2_copy = copy.copy(tokens_q2)
                    self.truncate_seq_tri(tokens_q_copy, tokens_q2_copy, tokens_r , self.max_seq_len - 3)
                    tokens_q_copy = tokens_q_copy + tokens_q2_copy

                tokens_qr = self.tokenizer.add_special_tokens_sentences_pair(tokens_q_copy, tokens_r)
                tokens_qr_arr.append(tokens_qr)
            entry["question"] = tokens_qr_arr

    def retokenize_and_convert_to_ids(self, _tokens, objects_replace_name):
        parsed_tokens = []
        for mixed_token in _tokens:
            if isinstance(mixed_token, list):
                tokens = [objects_replace_name[o] for o in mixed_token]
                retokenized_tokens = self.tokenizer.tokenize(tokens[0])
                for token, o in zip(tokens[1:], mixed_token[1:]):
                    retokenized_tokens.append('and')
                    
                    re_tokens = self.tokenizer.tokenize(token)
                    retokenized_tokens.extend(re_tokens)
                parsed_tokens.extend(retokenized_tokens)
            else:
                retokenized_tokens = self.tokenizer.tokenize(mixed_token)
                parsed_tokens.extend(retokenized_tokens)

        ids = self.tokenizer.convert_tokens_to_ids(parsed_tokens)
        return ids

    def load_raw_data_Q2A(self, cfg):
        datalist = []
        with open(os.path.join(self.anno_folder, self.stage + '.jsonl'), "rb") as f:
            for annotation in json_lines.reader(f):
                question = annotation["question"]
                image_id = int(annotation["img_id"].split("-")[1])
                anno_id = int(annotation["annot_id"].split("-")[1])
                if self.stage == "test":
                    ans_label = 0
                else:
                    ans_label = annotation["answer_label"]

                datalist.append({
                    "question": question,
                    "img_fn": annotation["img_fn"],
                    "objects":  annotation["objects"],
                    "answers": annotation["answer_choices"],
                    "metadata_fn": annotation["metadata_fn"],
                    "target": ans_label,
                    "image_id": image_id,
                    "anno_id": anno_id,
                })
        return datalist

    def load_raw_data_QA2R(self, cfg):
        datalist = []
        with open(os.path.join(self.anno_folder, self.stage + '.jsonl'), "rb") as f:
            for annotation in json_lines.reader(f):
                if self.stage == "test":
                    for answer in annotation["answer_choices"]:
                        question = annotation["question"] + ["[SEP]"] + answer
                        image_id = int(annotation["img_id"].split("-")[1])
                        anno_id = int(annotation["annot_id"].split("-")[1])
                        datalist.append({
                            "question": question,
                            "img_fn": annotation["img_fn"],
                            "objects":  annotation["objects"],
                            "answers": annotation["rationale_choices"],
                            "metadata_fn": annotation["metadata_fn"],
                            "target": 0,
                            "image_id": image_id,
                            "anno_id": anno_id,
                        })
                else:
                    question = annotation["question"]
                    ans_label = annotation["rationale_label"]
                    image_id = int(annotation["img_id"].split("-")[1])
                    anno_id = int(annotation["annot_id"].split("-")[1])
                    datalist.append({
                        "question": question,
                        "question_a": ["[SEP]"] + annotation["answer_choices"][annotation["answer_label"]],
                        "img_fn": annotation["img_fn"],
                        "objects":  annotation["objects"],
                        "answers": annotation["rationale_choices"],
                        "metadata_fn": annotation["metadata_fn"],
                        "target": ans_label,
                        "image_id": image_id,
                        "anno_id": anno_id,
                    })
        return datalist

    def load_raw_data(self, cfg):
        if self.task_name == "VCR_Q-A":
            return self.load_raw_data_Q2A(cfg)
        elif self.task_name == "VCR_QA-R":
            return self.load_raw_data_QA2R(cfg)
        else:
            raise ValueError(f"task_name should be VCR_Q-A or VCR_QA-R")

    def truncate_seq_pair(self, tokens_q, tokens_a, max_length):
        while len(tokens_a) + len(tokens_q) > max_length:
            if len(tokens_a) > len(tokens_q):
                tokens_a.pop()
            else:
                tokens_q.pop()

    def truncate_seq_tri(self, tokens_q, tokens_a, tokens_r, max_length):
        while len(tokens_q) + len(tokens_a) + len(tokens_r) > max_length:
            if len(tokens_r) > (len(tokens_q) + len(tokens_a)):
                tokens_r.pop()
            elif len(tokens_q) > 1:
                tokens_q.pop()
            else:
                tokens_a.pop()          

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        img_query = dataset_dict["metadata_fn"][:-5]

        prob = random.random()
        if prob > 0.5 and self.stage == 'train':
            image_path = os.path.join(self.feats_folder, img_query + ".npz")
            gt_image_path = os.path.join(self.gt_feat_folder, img_query + ".npz")
        else:
            image_path = os.path.join(self.feats_folder + "_mirror", img_query + ".npz")
            gt_image_path = os.path.join(self.gt_feat_folder + "_mirror", img_query + ".npz")
        features, image_locations = read_np_bbox(image_path, self.max_feat_num)
        gt_features, gt_image_locations = read_np_bbox(gt_image_path, self.max_feat_num)

        # merge two features.
        num_boxes = features.shape[0]
        gt_num_boxes = gt_features.shape[0]
        features[0] = (features[0] * num_boxes + gt_features[0] * gt_num_boxes) / (
            num_boxes + gt_num_boxes
        )

        # merge two boxes, and assign the labels.
        gt_boxes = gt_image_locations[1:gt_num_boxes]
        gt_features = gt_features[1:gt_num_boxes]
        gt_num_boxes = gt_num_boxes - 1

        gt_box_preserve = min(self.max_feat_num - 1, gt_num_boxes)
        gt_boxes = gt_boxes[:gt_box_preserve]
        gt_features = gt_features[:gt_box_preserve]
        gt_num_boxes = gt_box_preserve

        num_box_preserve = min(self.max_feat_num - gt_num_boxes, num_boxes)
        boxes = image_locations[:num_box_preserve]
        features = features[:num_box_preserve]

        # concatenate the boxes
        mix_boxes = np.concatenate((boxes, gt_boxes), axis=0)
        mix_features = np.concatenate((features, gt_features), axis=0)

        questions = [np.array(question).astype(np.int64) for question in dataset_dict["question"]]
        u_tokens_types = [np.array([0] * len(question)).astype(np.int64) for question in dataset_dict["question"]]

        ret = {
            kfg.IDS: str(dataset_dict["anno_id"]),
            kfg.SEQ_PER_SAMPLE: self.seq_per_img,
            kfg.ATT_FEATS: mix_features.astype('float32'),
            kfg.ATT_FEATS_LOC: mix_boxes.astype('float32'),
            kfg.U_TOKENS_IDS: questions,
            kfg.U_TOKENS_TYPE: u_tokens_types,
            kfg.U_TARGET_IDS: np.array([dataset_dict["target"]])
        }
        dict_as_tensor(ret)
        return ret

