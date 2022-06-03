# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import os
import sys
import tempfile
import json
from json import encoder
from xmodaler.config import kfg
from xmodaler.config import configurable
from .build import EVALUATION_REGISTRY

@EVALUATION_REGISTRY.register()
class COCOPrinter(object):
    def __init__(self, cfg, annfile, output_dir):
        super(COCOPrinter, self).__init__()
        self.output_dir = output_dir
        self.key = cfg.INFERENCE.ID_KEY
        self.value = cfg.INFERENCE.VALUE

    def eval(self, results, epoch):
        if self.output_dir is not None:
            fout = open(os.path.join(self.output_dir, 'results.txt'), 'a')
            for res in results:
                image_id = res[self.key]
                caption = res[self.value]
                fout.write('{}\t{}\n'.format(image_id, caption))
            fout.close()
        return results