# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import os
import sys
import pickle
import json
from json import encoder
from xmodaler.config import kfg
from xmodaler.config import configurable
from .build import EVALUATION_REGISTRY

@EVALUATION_REGISTRY.register()
class VQAEvaler(object):
    def __init__(self, cfg, annfile, output_dir):
        super(VQAEvaler, self).__init__()
        label2ans_path = os.path.join(cfg.DATALOADER.ANNO_FOLDER, "trainval_label2ans.pkl")
        self.label2ans = pickle.load(open(label2ans_path, "rb"))

        self.id2label = {}
        if len(annfile) > 0:
            answers_val = pickle.load(open(annfile, "rb"))
            for datum in answers_val:
                quesid = datum['question_id']
                self.id2label[quesid] = {}
                for i, label in enumerate(datum['labels']):
                    label_str = self.label2ans[label]
                    self.id2label[quesid][label_str] = datum['scores'][i]

        if output_dir is not None:
            self.output_dir = os.path.join(output_dir, 'results')
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir)
        else:
            self.output_dir = None

    def eval(self, results, epoch):
        for res in results:
            res['answer'] = self.label2ans[res['answer']]
        if self.output_dir is not None:
            json.dump(results, open(os.path.join(self.output_dir, str(epoch) + '.json'), "w"))

        accuracy = 0.
        for result in results:
            quesid = result['question_id']
            ans = result['answer']
            if quesid not in self.id2label:
                return { "accuracy": 0.0 }

            datum = self.id2label[quesid]
            if ans in datum:
                accuracy += datum[ans]

        accuracy = accuracy / len(results)
        return { "accuracy": accuracy }
