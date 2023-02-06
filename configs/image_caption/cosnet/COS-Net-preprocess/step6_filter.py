import os
import copy
import math
import tqdm
import numpy as np
import basic_utils as utils
from sklearn import metrics

min_words = 3
max_pos = 26
no_mension = 24
no_exist = 25

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def read_lines(filepath):
    with open(filepath, "r") as f:
        return [e.strip("\n") for e in f.readlines()]

def filter(clip_probs, ret_objs):
    top_k = 20
    obj_sel = set(np.argsort(clip_probs)[::-1][:top_k])
    
    ret_objs = set(ret_objs)
    ret_objs = obj_sel.intersection(ret_objs)

    objs_ids = list(set([o for o in ret_objs if clip_probs[o] > 0.8]))
    if len(objs_ids) < min_words:
        obj_sel = np.argsort(clip_probs)[::-1][:top_k]
        objs_ids = [o for o in obj_sel if o in ret_objs][:min_words]
    if len(objs_ids) < min_words:
        objs_ids = np.argsort(clip_probs)[::-1][:min_words]
    assert len(objs_ids) >= min_words

    objs_scores = [clip_probs[o] for o in objs_ids]
    return objs_ids, objs_scores


def generate():
    stage = 'train'
    data = utils.load_pickle('data/mscoco_caption_anno_' + stage + '.pkl')
    data2 = utils.load_pickle('data/mscoco_attr906_labels_' + stage + '.pkl')
    retrieval_res = utils.load_pickle('data/coco_image2text_retrieval_objs.pkl')  
    clip_probs = utils.load_pickle('data/mil_clip_coco_scores.pkl')

    count = 0
    avg_iou = 0.0
    avg_cor = 0.0
    avg_hit = 0.0
    vocab = read_lines('data/new_vocab_words.txt')
    avg_missing = 0
    avg_clip_len = 0

    res = []
    for ent, ent2 in tqdm.tqdm(zip(data, data2)):
        count = count + 1
        image_id = ent['image_id']
        full_gt_objs = set(np.where(ent2['labels'] > 0)[0])
        gt_objs = ent2['gt_objs']

        clip_prob = clip_probs[image_id]
        ret_objs = retrieval_res[image_id]
    
        clip_objs, objs_scores = filter(clip_prob, ret_objs)

        correct_rate = len(full_gt_objs.intersection(clip_objs)) * 1.0 / (len(clip_objs) + 1e-10)
        hit_rate = max([ len(e.intersection(clip_objs)) * 1.0 / len(e) for e in gt_objs ])
        iou_rate = max([ len(e.intersection(clip_objs)) * 1.0 / len(e.union(clip_objs)) for e in gt_objs ] )
        
        avg_iou += iou_rate
        avg_cor += correct_rate
        avg_hit += hit_rate

        clip_objs = np.array(clip_objs)      # final pred: bg -- len(vocab)
        objs_scores = np.array(objs_scores)  
        clip_objs_labels = np.zeros((len(clip_objs), )).astype(np.int64) - 1
        missing_labels = []
        avg_clip_len += len(clip_objs)

        for i in range(len(clip_objs)):
            if clip_objs[i] in full_gt_objs:
                clip_objs_labels[i] = clip_objs[i]
            else:
                clip_objs_labels[i] = len(vocab)

        pred_set = set(clip_objs)
        for gt_attr in full_gt_objs:
            if gt_attr not in pred_set:
                missing_labels.append(gt_attr)
        avg_missing += len(missing_labels)

        tmp = {
            'image_id': image_id,
            'attr_pred': clip_objs,
            'attr_labels': clip_objs_labels,
            'missing_labels': missing_labels
        }
        if 'tokens_ids' in ent:
            tmp.update({'tokens_ids': ent['tokens_ids']})
        if 'target_ids' in ent:
            tmp.update({'target_ids': ent['target_ids']})
        res.append(tmp)

    avg_iou /= count
    avg_cor /= count
    avg_hit /= count
    avg_clip_len /= count
    avg_missing /= count
    print('average iou: ' + str(avg_iou))
    print('average cor: ' + str(avg_cor))
    print('average hit: ' + str(avg_hit))
    print('avg missing: ' + str(avg_missing))
    print('avg_clip_len: ' + str(avg_clip_len))
    utils.save_pickle(res, 'mscoco_caption_anno_clipfilter_' + stage + '.pkl')
    

if __name__ == "__main__":
    generate()
