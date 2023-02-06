import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import clip
import tqdm
import math
import pickle
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomCrop, FiveCrop

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float()

def resize_pos_embed(posemb, posemb_new):
    '''
    resize position embedding with class token
    example: 224:(14x14+1)-> 384: (24x24+1)
    return: new position embedding
    '''
    ntok_new = posemb_new.shape[0]

    posemb_grid = posemb  # posemb_tok is for cls token, posemb_grid for the following tokens

    gs_old = int(math.sqrt(len(posemb_grid)))  # 14
    gs_new = int(math.sqrt(ntok_new))  # 24
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(
        0, 3, 1, 2)  # [1, 196, dim]->[1, 14, 14, dim]->[1, dim, 14, 14]
    posemb_grid = F.interpolate(
        posemb_grid, size=(gs_new, gs_new),
        mode='bicubic')  # [1, dim, 14, 14] -> [1, dim, 24, 24]
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(
        1, gs_new * gs_new, -1)  # [1, dim, 24, 24] -> [1, 24*24, dim]
    return posemb_grid.squeeze(0)

class ClipRetrieval(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_seq_len = 25
        self.num_classes = 906
        clip_model, preprocess = clip.load("ViT-B/16", jit=False)
        convert_models_to_fp32(clip_model)

        self.mil = nn.Linear(512, self.num_classes)
        self.gate = nn.Sigmoid()

        for block in clip_model.transformer.resblocks:
            att_mask = block.attn_mask
            att_mask = att_mask[:self.max_seq_len, :self.max_seq_len]
            block.attn_mask = att_mask
        positional_embedding = clip_model.positional_embedding[:self.max_seq_len]
        clip_model.positional_embedding = nn.Parameter(positional_embedding)
        self.clip_model = clip_model

        n_px = (224, 224)
        self.transform = Compose([
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def encode_text(self, text_inputs):
        text_features = self.clip_model.encode_text(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clipmodel = ClipRetrieval().cuda()
    state_dict = torch.load('data/retrieval_model.pth', map_location="cpu")['trainer']['ema']
    new_dict = {}
    for k in state_dict:
        new_dict[k[14:]] = state_dict[k]
    state_dict = new_dict
    clipmodel.load_state_dict(state_dict)

    print('encode text')
    with open('data/coco_sents_refine.txt') as fid:
        full_sents = ['a photo of ' + line.strip() for line in fid]

    feat_sents = np.zeros((len(full_sents), 512)).astype('float32')
    batch_size = 64
    num_batches = len(full_sents) // batch_size + 1

    with torch.no_grad():
        for i in tqdm.tqdm(range(num_batches)):
            if i == num_batches - 1:
                sents = full_sents[i*batch_size:]
            else:
                sents = full_sents[i*batch_size:i*batch_size+batch_size]

            text = clip.tokenize(sents).to(device)
            text = text[:, :clipmodel.max_seq_len]
            for j in range(text.shape[0]):
                if text[j, -1] != 0:
                    text[j, -1] = 49407

            text_features = clipmodel.encode_text(text)
            text_features = text_features.data.cpu().numpy()
            feat_sents[i*batch_size:i*batch_size + len(text_features)] = text_features

    save_pickle(feat_sents, 'data/clip_coco_sents.pkl')

