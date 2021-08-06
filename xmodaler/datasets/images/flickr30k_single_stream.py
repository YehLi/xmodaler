# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import copy
import sys
import torch
import random
import numpy as np
from collections import defaultdict

from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import dict_as_tensor, flat_list_of_lists, pad_tensor
from .flickr30k import Flickr30kDataset
from ..build import DATASETS_REGISTRY

__all__ = ["Flickr30kDatasetForSingleStream", "Flickr30kDatasetForSingleStreamVal"]


def get_max_len_from_mask(mask):
    return int(mask.sum(1).max().item())


def clip_v_inputs(v_feats, spatials, image_mask):
    max_len = get_max_len_from_mask(image_mask)
    v_feats = v_feats[:, :max_len]
    spatials = spatials[:, :max_len]
    image_mask = image_mask[:, :max_len]
    return v_feats, spatials, image_mask


@DATASETS_REGISTRY.register()
class Flickr30kDatasetForSingleStream(Flickr30kDataset):
    @configurable
    def __init__(
        self,
        stage: str,
        anno_folder: str,
        anno_file: str,
        feats_folder: str,
        max_feat_num: int,
        max_seq_len: int,
        use_global_v: bool,
        negative_size: int,
        tokenizer,
        cfg
    ):
        super(Flickr30kDatasetForSingleStream, self).__init__(
            stage,
            anno_folder,
            anno_file,
            feats_folder,
            max_feat_num,
            max_seq_len,
            use_global_v,
            tokenizer
        )
        self.negative_size = negative_size

        # load img_ids for neg sample
        datalist = self.load_data(cfg)
        self.imgid2caps = defaultdict(set)
        self.cap2imgids = defaultdict(set)

        for item in datalist:
            image_id = item['image_id']
            caption = tuple(item['captions']) # NOTE: actually it is one caption

            self.imgid2caps[image_id].add(caption)
            self.cap2imgids[caption].add(image_id)

        self.imgid2caps = {k:list(v) for k,v in dict(self.imgid2caps).items()}
        self.cap2imgids = {k:list(v) for k,v in dict(self.cap2imgids).items()}
        self.image_ids_set = set(list(self.imgid2caps.keys()))

    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        ret = super().from_config(cfg, stage)
        ret['negative_size'] = cfg.DATALOADER.NEGATIVE_SIZE
        ret['cfg'] = cfg
        return ret

    def sample_neg_pairs(self, pos_meta_data, neg_meta_data_list):
        (image_id, features1, image_locations1, caption1, u_tokens_type1) = pos_meta_data
        neg_features_list, neg_image_locations_list, neg_caption_list, neg_u_tokens_type_list = neg_meta_data_list

        black_img_ids = flat_list_of_lists([self.cap2imgids[c] for c in self.imgid2caps[image_id]])
        image_id_pool = list(self.image_ids_set - set(black_img_ids))
        
        # sample a cap wrong
        img_id2 = random.choice(image_id_pool)
        features2, image_locations2 = features1, image_locations1
        caption2 = random.choice(self.imgid2caps[img_id2])
        caption2, u_tokens_type2 = self.format_cap(caption2)

        # sample an img wrong
        img_id3 = random.choice(image_id_pool)
        features3, image_locations3 = self.load_img_feat(img_id3)
        caption3, u_tokens_type3 = caption1, u_tokens_type1

        # add neg sample
        neg_features_list.extend([features2, features3])
        neg_image_locations_list.extend([image_locations2, image_locations3])
        neg_caption_list.extend([caption2, caption3])
        neg_u_tokens_type_list.extend([u_tokens_type2, u_tokens_type3])

    def __call__(self, dataset_dict):
        assert self.stage == 'train'
        dataset_dict = copy.deepcopy(dataset_dict)
        image_id = dataset_dict['image_id']
        
        # Positive
        features1, image_locations1 = self.load_img_feat(image_id)
        caption1 = dataset_dict['captions']
        caption1, u_tokens_type1 = self.format_cap(caption1)

        pos_meta_data = (image_id, features1, image_locations1, caption1, u_tokens_type1)

        neg_features_list = []
        neg_image_locations_list = []
        neg_caption_list = []
        neg_u_tokens_type_list = []
        neg_meta_data_list = [neg_features_list, neg_image_locations_list, neg_caption_list, neg_u_tokens_type_list]
        for _ in range(self.negative_size):
            # negative samples.
            # 1: correct one, 2: random caption wrong, 3: random image wrong.
            # self.negative_size pair <==> 2*self.negative_size negatives pair
            self.sample_neg_pairs(pos_meta_data, neg_meta_data_list)

        features = [features1] + neg_features_list
        image_locations = [image_locations1] + neg_image_locations_list
        captions = [caption1] + neg_caption_list
        u_tokens_type = [u_tokens_type1] + neg_u_tokens_type_list

        ret = {
            kfg.ATT_FEATS: [x.astype('float32') for x in features],
            kfg.ATT_FEATS_LOC: [x.astype('float32') for x in image_locations],
            kfg.U_TOKENS_IDS: captions,
            kfg.U_TOKENS_TYPE: u_tokens_type,
            kfg.U_TARGET_IDS: np.array([0], dtype=np.int64).reshape(-1, 1)
        }

        dict_as_tensor(ret)
        ret[kfg.SAMPLE_PER_SAMPLE] = len(features)
        return ret


@DATASETS_REGISTRY.register()
class Flickr30kDatasetForSingleStreamVal(Flickr30kDataset):
    @configurable
    def __init__(
        self,
        stage: str,
        anno_folder: str,
        anno_file: str,
        feats_folder: str,
        max_feat_num: int,
        max_seq_len: int,
        use_global_v: bool,
        inf_batch_size: int,
        tokenizer,
        cfg
    ):
        super(Flickr30kDatasetForSingleStreamVal, self).__init__(
            stage,
            anno_folder,
            anno_file,
            feats_folder,
            max_feat_num,
            max_seq_len,
            use_global_v,
            tokenizer
        )
        self.inf_batch_size = inf_batch_size

        # load img_ids for neg sample
        datalist = super().load_data(cfg)
        self.imgid2caps = {item['image_id']:item['captions'] for item in datalist}
        self.all_img_ids = list(self.imgid2caps.keys())
        self.imgid2featidx = {i:j for j,i in enumerate(self.all_img_ids)}

        tid2imgid = {}
        imgid2tids = defaultdict(list)

        caption_all = []
        tid = 0
        for image_id, captions in self.imgid2caps.items():
            sent_num = len(captions)
            for i, caption in enumerate(captions):
                curr_tid = tid + i
                caption_all.append(caption)
                tid2imgid[curr_tid] = image_id
                imgid2tids[image_id].append(curr_tid)
            tid += sent_num

        self.tid2imgid = tid2imgid
        self.imgid2tids = dict(imgid2tids)
        self.caption_all = caption_all
        
        # load v_feature pool
        features_all = []
        image_locations_all = []
        for image_id, feat_idx in self.imgid2featidx.items():
            features, image_locations = self.load_img_feat(image_id)
            features_all.append(torch.as_tensor(features).float())
            image_locations_all.append(torch.as_tensor(image_locations).float())
            
            sys.stdout.write('%d/%d\r' % (feat_idx, len(self.all_img_ids)))
            sys.stdout.flush()
        
        vfeats_all, vmasks_all = pad_tensor(features_all, padding_value=0, use_mask=True)
        img_loc_all = pad_tensor(image_locations_all, padding_value=0, use_mask=False)
        self.features_all = vfeats_all.float()
        self.image_mask_all = vmasks_all.float()
        self.image_locations_all = img_loc_all.float()

    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        ret = super().from_config(cfg, stage)
        ret['inf_batch_size'] = cfg.DATALOADER.INF_BATCH_SIZE
        ret['cfg'] = cfg
        return ret

    def load_data(self, cfg):
        # sample by text
        datalist = []
        for tid, caption in enumerate(self.caption_all):
            datalist.append({
                'tid': tid,
                'caption': caption,
                'tid2imgid': self.tid2imgid[tid],
                'imgid2tids': tuple(self.imgid2tids[self.tid2imgid[tid]]),
                'total_img_num': len(self.all_img_ids)
            })
        # datalist = datalist[:20] # for debug
        return datalist

    def __call__(self, dataset_dict):
        assert self.stage != 'train'
        dataset_dict = copy.deepcopy(dataset_dict)
        
        # NOTE: Only support text->image retrieval now
        tid = dataset_dict['tid']
        matched_imgid = dataset_dict['tid2imgid']
        matched_imgfeatidx = self.imgid2featidx[matched_imgid]
        caption = dataset_dict['caption']
        total_img_num = dataset_dict['total_img_num']

        # prepare txt pool
        caption, u_tokens_type = self.format_cap(caption)
        tokens_masks = np.array([1] * len(caption), dtype=np.int64)
        
        u_tokens_ids = torch.tensor(caption).long()
        u_tokens_type = torch.tensor(u_tokens_type).long()
        tokens_masks = torch.tensor(tokens_masks).long()

        u_tokens_ids_pool = u_tokens_ids.unsqueeze(0).expand(total_img_num, -1)
        u_tokens_type_pool = u_tokens_type.unsqueeze(0).expand(total_img_num, -1)
        tokens_masks_pool = tokens_masks.unsqueeze(0).expand(total_img_num, -1)

        # prepare img pool
        img_feats_pool, image_locations_pool, image_mask_pool = self.features_all.clone(), self.image_locations_all.clone(), self.image_mask_all.clone()

        # chunk to minibatch
        u_tokens_ids_pool = torch.split(u_tokens_ids_pool, self.inf_batch_size, dim=0)
        u_tokens_type_pool = torch.split(u_tokens_type_pool, self.inf_batch_size, dim=0)
        tokens_masks_pool = torch.split(tokens_masks_pool, self.inf_batch_size, dim=0)
        img_feats_pool = torch.split(img_feats_pool, self.inf_batch_size, dim=0)
        image_locations_pool = torch.split(image_locations_pool, self.inf_batch_size, dim=0)
        image_mask_pool = torch.split(image_mask_pool, self.inf_batch_size, dim=0)

        # preprocess in the dataset
        batches = []
        for u_tokens_ids, u_tokens_type, tokens_masks, img_feats, image_locations, image_mask in \
            zip(u_tokens_ids_pool, u_tokens_type_pool, tokens_masks_pool, img_feats_pool, image_locations_pool, image_mask_pool):

            img_feats, image_locations, image_mask = clip_v_inputs(img_feats, image_locations, image_mask)

            batch = {
                kfg.ATT_FEATS: img_feats,
                kfg.ATT_FEATS_LOC: image_locations,
                kfg.ATT_MASKS: image_mask,

                kfg.U_TOKENS_IDS: u_tokens_ids,
                kfg.TOKENS_MASKS: tokens_masks,
                kfg.U_TOKENS_TYPE: u_tokens_type,
            }
            batch['matched_imgfeatidx'] = matched_imgfeatidx
            batch['total_img_num'] = total_img_num
            dict_as_tensor(batch)

            batches.append(batch)

        return batches