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
    def __init__(self, cfg, annfile):
        super(VQAEvaler, self).__init__()
        answers_val = pickle.load(open(annfile, "rb"))
        label2ans_path = os.path.join(cfg.DATALOADER.ANNO_FOLDER, "trainval_label2ans.pkl")
        self.label2ans = pickle.load(open(label2ans_path, "rb"))

        self.id2label = {}
        for datum in answers_val:
                quesid = datum['question_id']
                self.id2label[quesid] = {}
                for i, label in enumerate(datum['labels']):
                    label_str = self.label2ans[label]
                    self.id2label[quesid][label_str] = datum['scores'][i]

    def eval(self, results):
        if 'answer' not in results[0]:
            return { "accuracy": 0.0 }

        accuracy = 0.
        for result in results:
            quesid = result['question_id']
            ans = result['answer']
            datum = self.id2label[quesid]
            if ans in datum:
                accuracy += datum[ans]

        accuracy = accuracy / len(results)
        return { "accuracy": accuracy }
