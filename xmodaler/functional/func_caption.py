# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import os

def decode_sequence(vocab, seq):
    N, T = seq.size()
    sents = []
    for n in range(N):
        words = []
        for t in range(T):
            ix = seq[n, t]
            if ix == 0:
                break
            words.append(vocab[ix])
        sent = ' '.join(words)
        sents.append(sent)
    return sents

def decode_sequence_bert(tokenizer, seq, sep_token_id):
    N, T = seq.size()
    seq = seq.data.cpu().numpy()
    sents = []
    for n in range(N):
        words = []
        for t in range(T):
            ix = seq[n, t]
            if ix == sep_token_id:
                break
            words.append(tokenizer.ids_to_tokens[ix])
        sent = tokenizer.convert_tokens_to_string(words)
        sents.append(sent)
    return sents