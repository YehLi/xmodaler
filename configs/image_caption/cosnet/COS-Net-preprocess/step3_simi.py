import os
import numpy as np
import torch
import tqdm
import pickle

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

top_k = 20
image_features = load_pickle('data/clip_coco_imgs.pkl')
text_features = load_pickle('data/clip_coco_sents.pkl')
text_features = torch.as_tensor(text_features).cuda()

res = {}
image_ids_list = []
imgfeats = []

for image_id in tqdm.tqdm(image_features):
    imgfeats.append(torch.as_tensor(image_features[image_id]).cuda())
    image_ids_list.append(image_id)
imgfeats = torch.stack(imgfeats, dim=0)

batch_size = 64
num_batches = len(image_ids_list) // batch_size + 1

for i in tqdm.tqdm(range(num_batches)):
    if i == num_batches - 1:
        cur_feats = imgfeats[i*batch_size:, ]
        cur_imgids = image_ids_list[i*batch_size:]
    else:
        cur_feats = imgfeats[i*batch_size:i*batch_size+batch_size]
        cur_imgids = image_ids_list[i*batch_size:i*batch_size+batch_size]

    logits_per_image = cur_feats @ text_features.t()
    sents_ids = torch.argsort(logits_per_image, dim=-1, descending=True)[:, :top_k]
    sents_ids = sents_ids.data.cpu().numpy()
    logits_per_image = logits_per_image.data.cpu().numpy()
    
    for image_id, sents_id, simi_scores in zip(cur_imgids, sents_ids, logits_per_image):
        scores = simi_scores[sents_id]
        res[image_id] = (sents_id, scores)

print(len(res))
save_pickle(res, 'data/coco_image2text_retrieval.pkl')
