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
class VCREvaler(object):
    def __init__(self, cfg, annfile, output_dir):
        super(VCREvaler, self).__init__()

    def eval(self, results_list, epoch):
        q2a_results = results_list[0]
        qa2r_results = results_list[1]

        q2a_res = {}
        q2a_accuracy = 0
        for res in q2a_results:
            question_id = res['question_id']
            answer = res['answer']
            target = res[kfg.U_TARGET_IDS]

            if answer == target:
                q2a_accuracy += 1
                q2a_res[question_id] = True
            else:
                q2a_res[question_id] = False
        q2a_accuracy /= len(q2a_results)

        qa2r_res = {}
        qa2r_accuracy = 0
        for res in qa2r_results:
            question_id = res['question_id']
            answer = res['answer']
            target = res[kfg.U_TARGET_IDS]

            if answer == target:
                qa2r_accuracy += 1
                qa2r_res[question_id] = True
            else:
                qa2r_res[question_id] = False
        qa2r_accuracy /= len(qa2r_results)

        accuracy = 0
        for qid in q2a_res:
            if q2a_res[qid] == True and qa2r_res[qid] == True:
                accuracy += 1
        accuracy /= len(q2a_res)

        return {
            'Q -> A': q2a_accuracy,
            'QA -> R': qa2r_accuracy,
            'Q -> AR': accuracy
        }