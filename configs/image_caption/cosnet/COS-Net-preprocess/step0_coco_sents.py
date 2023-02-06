import os
import numpy as np
import basic_utils as utils

if __name__ == "__main__":
    coco_ids = utils.read_lines('data/coco_train_image_id.txt')
    coco_ids = set(coco_ids)

    full_sents = []
    raw_anno = utils.load_json('data/dataset_coco.json')['images']
    for data in raw_anno:
        image_id = str(data['cocoid'])

        if image_id not in coco_ids:
            continue
        sents = data['sentences']
        for sent in sents:
            sent = sent['raw'].lower().strip(' ').strip('.')
            sent = sent.replace('\n', ' ')
            sent = sent.replace(';', '')
            words = sent.split(' ')[0:20]
            sent = ' '.join(words)
            full_sents.append(sent)

    print(len(full_sents))
    utils.save_lines(full_sents, 'coco_sents_refine.txt')
