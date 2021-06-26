import os
import sys
import tempfile
import json
from json import encoder
from xmodaler.config import kfg
from xmodaler.config import configurable
from .build import EVALUATION_REGISTRY

sys.path.append(kfg.COCO_PATH)
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

@EVALUATION_REGISTRY.register()
class COCOEvaler(object):
    def __init__(self, cfg, annfile):
        super(COCOEvaler, self).__init__()
        self.coco = COCO(annfile)
        if not os.path.exists(kfg.TEMP_DIR):
            os.mkdir(kfg.TEMP_DIR)

    def eval(self, result):
        in_file = tempfile.NamedTemporaryFile(mode='w', delete=False, dir=kfg.TEMP_DIR)
        json.dump(result, in_file)
        in_file.close()

        cocoRes = self.coco.loadRes(in_file.name)
        cocoEval = COCOEvalCap(self.coco, cocoRes)
        cocoEval.evaluate()
        os.remove(in_file.name)
        return cocoEval.eval