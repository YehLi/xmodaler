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
import numpy as np
from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import read_np, dict_as_tensor, boxes_to_locfeats, read_np_bbox
from xmodaler.tokenization import BertTokenizer
from ..build import DATASETS_REGISTRY

__all__ = ["VQADataset"]

@DATASETS_REGISTRY.register()
class VQADataset:
    @configurable
    def __init__(
        self,
        stage: str,
        anno_folder: str,
        ans2label_path: str,
        label2ans_path: str,
        feats_folder: str,
        max_feat_num: int,
        max_seq_len: int,
        tokenizer
    ):
        self.stage = stage
        self.anno_folder = anno_folder
        self.ans2label = pickle.load(open(ans2label_path, "rb"))
        self.label2ans = pickle.load(open(label2ans_path, "rb"))
        self.feats_folder = feats_folder
        self.max_feat_num = max_feat_num
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.num_labels = len(self.ans2label)

        
    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        ans2label_path = os.path.join(cfg.DATALOADER.ANNO_FOLDER, "trainval_ans2label.pkl")
        label2ans_path = os.path.join(cfg.DATALOADER.ANNO_FOLDER, "trainval_label2ans.pkl")
        
        feats_folder = cfg.DATALOADER.FEATS_FOLDER
        if stage == "test":
            feats_folder = feats_folder + "_test2015"

        ret = {
            "stage": stage,
            "anno_folder": cfg.DATALOADER.ANNO_FOLDER,
            "ans2label_path": ans2label_path,
            "label2ans_path": label2ans_path,
            "feats_folder": feats_folder,
            "max_feat_num": cfg.DATALOADER.MAX_FEAT_NUM,
            "max_seq_len": cfg.MODEL.MAX_SEQ_LEN,
            "tokenizer": BertTokenizer.from_pretrained(cfg.MODEL.PRETRAINING.MODEL_NAME,
                do_lower_case=cfg.MODEL.PRETRAINING.DO_LOWER_CASE),
        }
        return ret

    def load_data(self, cfg):
        cache_path = os.path.join(
            self.anno_folder, "cache", 
            "VQA_%s_%d.pkl" % (self.stage, self.max_seq_len)
        )
        if not os.path.exists(cache_path):
            datalist = self.load_raw_data(cfg)    
            self.tokenize(datalist)
            pickle.dump(datalist, open(cache_path, "wb"))
        datalist = pickle.load(open(cache_path, "rb"))
        return datalist

    def tokenize(self, datalist):
        for entry in datalist:
            tokens = self.tokenizer.encode(entry["question"])
            tokens = tokens[: self.max_seq_len - 2]
            tokens = self.tokenizer.add_special_tokens_single_sentence(tokens)
            entry["question"] = tokens

    def load_raw_data(self, cfg):
        if self.stage == 'train': # trainval mode
            question_path_train = os.path.join(self.anno_folder, "v2_OpenEnded_mscoco_train2014_questions.json")
            questions_train = sorted(
                json.load(open(question_path_train))["questions"],
                key=lambda x: x["question_id"],
            )
            answer_path_train = os.path.join(self.anno_folder, "train_target.pkl")
            answers_train = pickle.load(open(answer_path_train, "rb"))
            answers_train = sorted(answers_train, key=lambda x: x["question_id"])

            question_path_val = os.path.join(self.anno_folder, "v2_OpenEnded_mscoco_val2014_questions.json")
            questions_val = sorted(
                json.load(open(question_path_val))["questions"],
                key=lambda x: x["question_id"],
            )   
            answer_path_val = os.path.join(self.anno_folder, "val_target.pkl")
            answers_val = pickle.load(open(answer_path_val, "rb"))
            answers_val = sorted(answers_val, key=lambda x: x["question_id"])

            # VG
            vg_question_path_train = os.path.join(self.anno_folder, "VG_questions2.json")
            vg_questions_train = sorted(
                json.load(open(vg_question_path_train))["questions"],
                key=lambda x: x["question_id"],
            )
            vg_answer_path_train = os.path.join(self.anno_folder, "vg_target.pkl")
            vg_answers_train = pickle.load(open(vg_answer_path_train, "rb"))
            vg_answers_train = sorted(vg_answers_train, key=lambda x: x["question_id"])

            questions = questions_train + questions_val[:-3000] + vg_questions_train
            answers = answers_train + answers_val[:-3000] + vg_answers_train
        elif self.stage == "val": # minval
            question_path_val = os.path.join(self.anno_folder, "v2_OpenEnded_mscoco_val2014_questions.json")
            questions_val = sorted(
                json.load(open(question_path_val))["questions"],
                key=lambda x: x["question_id"],
            )
            answer_path_val = os.path.join(self.anno_folder, "val_target.pkl")
            answers_val = pickle.load(open(answer_path_val, "rb"))
            answers_val = sorted(answers_val, key=lambda x: x["question_id"])
            questions = questions_val[-3000:]
            answers = answers_val[-3000:]
        else:
            question_path_test = os.path.join(self.anno_folder, "v2_OpenEnded_mscoco_test2015_questions.json")
            questions_test = sorted(
                json.load(open(question_path_test))["questions"],
                key=lambda x: x["question_id"],
            )
            questions = questions_test

        datalist = []
        if self.stage == "test":
            for question in questions:
                datalist.append({
                    "question_id": str(question["question_id"]),
                    "image_id": str(question["image_id"]),
                    "question": question["question"],
                })
        else:
            assert len(questions) == len(answers)
            for question, answer in zip(questions, answers):
                assert question["question_id"] == answer["question_id"]
                assert question["image_id"] == answer["image_id"]
                
                answer.pop("image_id")
                answer.pop("question_id")
                datalist.append({
                    "question_id": str(question["question_id"]),
                    "image_id": str(question["image_id"]),
                    "question": question["question"],
                    "answer": answer,
                })
        return datalist

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image_id = dataset_dict['image_id']
        question_id = dataset_dict["question_id"]
        
        image_path = os.path.join(self.feats_folder, image_id + ".npz")
        features, image_locations = read_np_bbox(image_path, self.max_feat_num)

        question = np.array(dataset_dict["question"])
        u_tokens_type = np.array([0] * len(question))

        ret = {
            kfg.IDS: question_id,
            kfg.ATT_FEATS: features.astype('float32'),
            kfg.ATT_FEATS_LOC: image_locations.astype('float32'),
            kfg.U_TOKENS_IDS: question.astype(np.int64),
            kfg.U_TOKENS_TYPE: u_tokens_type.astype(np.int64),
        }

        if "answer" in dataset_dict:
            answer = dataset_dict["answer"]
            labels = answer["labels"]
            scores = answer["scores"]

            target = np.zeros(self.num_labels)
            if len(labels) > 0:
                for label, score in zip(labels, scores):
                    target[label] = score
            ret.update({ kfg.U_TARGET_IDS: target.astype('float32') })
        dict_as_tensor(ret)
        return ret