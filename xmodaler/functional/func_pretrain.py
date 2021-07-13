# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import random
import numpy as np

def random_word(tokens, tokenizer):
    output_labels = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability

        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = np.random.randint(len(tokenizer))
                # torch.randint(len(tokenizer), labels.shape, dtype=torch.long)

            # -> rest 10% randomly keep current token
            # append current token to output (we will predict these later)
            output_labels.append(token)
        else:
            # no masking token (will be ignored by loss function later)
            output_labels.append(-1)
    return tokens, output_labels

def random_region(image_feats, overlaps):
    output_labels = []
    masked_labels = np.zeros((image_feats.shape[0]))

    num_boxes = overlaps.shape[0]
    for i in range(num_boxes):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15
            if prob < 0.9:
                image_feats[i] = 0
            # mask the overlap regions into zeros
            masked_labels = np.logical_or(masked_labels, overlaps[i] > 0.4)
            output_labels.append(1)
        else:
            output_labels.append(-1)

    masked_labels = [idx for idx, item in enumerate(masked_labels) if item]
    if masked_labels:
        image_feats[masked_labels, :] = 0
    masked_num = len(masked_labels)

    return image_feats, output_labels, masked_num

def caption_to_mask_tokens(caption, max_seq_length, tokenizer):
    tokens_ids = tokenizer.encode(caption)
    tokens_ids = tokens_ids[: max_seq_length - 2]
    g_tokens_labels = tokens_ids
    g_tokens_labels = tokenizer.add_special_tokens_single_sentence(g_tokens_labels)
    g_tokens_labels = g_tokens_labels[1:] + [-1]

    tokens_ids, u_tokens_labels = random_word(tokens_ids, tokenizer)
    u_tokens_labels = [-1] + u_tokens_labels + [-1]
    tokens_ids = tokenizer.add_special_tokens_single_sentence(tokens_ids)
    tokens_ids = np.array(tokens_ids)
    u_tokens_labels = np.array(u_tokens_labels)
    g_tokens_labels = np.array(g_tokens_labels)
    return tokens_ids, u_tokens_labels, g_tokens_labels