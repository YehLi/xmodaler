import os
import copy
import tqdm
import numpy as np
import basic_utils as utils

def read_lines(filepath):
    with open(filepath, "r") as f:
        return [e.strip("\n") for e in f.readlines()]

if __name__ == "__main__":
    stage = 'train'
    coco_ids = utils.read_lines(os.path.join('data', 'coco_' + stage + '_image_id.txt'))
    coco_ids = set(coco_ids)
    raw_anno = utils.load_json('data/dataset_coco.json')['images']
    vocab = read_lines('data/new_vocab_words.txt')
    obj_dict = {w:i for i,w in enumerate(vocab)}

    res = []
    for ent in tqdm.tqdm(raw_anno):
        image_id = str(ent['cocoid'])
        if image_id not in coco_ids:
            continue
        labels = np.zeros((len(vocab), )).astype(np.int32)

        gt_objs = []
        for sent in ent['sentences']:
            gto = []
            for w in sent['tokens']:
                if w in obj_dict:
                    labels[obj_dict[w]] = 1
                    gto.append(obj_dict[w])
            if len(gto) > 0:
               gt_objs.append(set(gto))

        res.append({
            'image_id': image_id,
            'labels': labels,
            'gt_objs': gt_objs,
            "filename": ent["filename"],
            "filepath": ent["filepath"]
        })

    print(len(res))
    utils.save_pickle(res, 'mscoco_attr906_labels_' + stage + '.pkl')


