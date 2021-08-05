"""
From original at https://github.com/ChenRocks/UNITER/blob/master/utils/distributed.py
Original copyright of Microsoft code below, modifications by Jianjie Luo, Copyright 2021.	
"""

import math
import pickle

import torch
import torch.distributed as dist


def broadcast_tensors(tensors, root_rank, buffer_size=10485760):
    """broadcast tensors in chunks of the specified size.

    Args:
        tensors: list of Tensors to broadcast
        root_rank: rank to broadcast
        buffer_size: broadcast chunk size in bytes
    """
    # buffer size in bytes, determine equiv. # of elements based on data type
    buffer_t = tensors[0].new(math.ceil(buffer_size / tensors[0].element_size())).zero_()
    buffer = []

    def broadcast_buffer():
        # copy tensors into buffer_t
        offset = 0
        for t in buffer:
            numel = t.numel()
            buffer_t[offset:offset+numel].copy_(t.view(-1))
            offset += numel

        # broadcast
        dist.broadcast(buffer_t[:offset], root_rank)

        # copy all-reduced buffer back into tensors
        offset = 0
        for t in buffer:
            numel = t.numel()
            t.view(-1).copy_(buffer_t[offset:offset+numel])
            offset += numel

    filled = 0
    for t in tensors:
        sz = t.numel() * t.element_size()
        if sz > buffer_size:
            # tensor is bigger than buffer, broadcast directly
            dist.broadcast(t, root_rank)

        elif filled + sz > buffer_size:
            # buffer is full, broadcast and replace buffer with tensor
            broadcast_buffer()
            buffer = [t]
            filled = sz
        else:
            # add tensor to buffer
            buffer.append(t)
            filled += sz

    if len(buffer) > 0:
        broadcast_buffer()


def _encode(enc, max_size, use_max_size=False):
    enc_size = len(enc)
    enc_byte = max(math.floor(math.log(max_size, 256)+1), 1)
    if use_max_size:
        # this is used for broadcasting
        buffer_ = torch.cuda.ByteTensor(max_size+enc_byte)
    else:
        buffer_ = torch.cuda.ByteTensor(enc_size+enc_byte)
    remainder = enc_size
    for i in range(enc_byte):
        base = 256 ** (enc_byte-i-1)
        buffer_[i] = remainder // base
        remainder %= base
    buffer_[enc_byte:enc_byte+enc_size] = torch.ByteTensor(list(enc))
    return buffer_, enc_byte


def _decode(buffer_, enc_byte):
    size = sum(256 ** (enc_byte-i-1) * buffer_[i].item() for i in range(enc_byte))
    bytes_list = bytes(buffer_[enc_byte:enc_byte+size].tolist())
    shift = size + enc_byte
    return bytes_list, shift


_BUFFER_SIZE = 4096


def all_gather_list(data):
    """Gathers arbitrary data from all nodes into a list."""
    n_gpu = torch.cuda.device_count()

    enc = pickle.dumps(data)

    tensor_list = [torch.zeros(1, dtype=torch.int64).cuda() for _ in range(n_gpu)]
    enc_size = len(enc)
    dist.all_gather(tensor_list, tensor=torch.tensor([enc_size]).cuda())
    max_size = torch.cat(tensor_list, dim=0).view(-1).max().item()
    in_buffer, enc_byte = _encode(enc, max_size)

    out_buffer = [in_buffer.new_zeros(in_buffer[:enc_byte+enc_size].shape) for _ in range(n_gpu)]
    dist.all_gather(out_buffer, tensor=in_buffer[:enc_byte+enc_size])
    out_buffer = torch.cat(out_buffer, dim=0)

    results = []
    for _ in range(n_gpu):
        bytes_list, shift = _decode(out_buffer, enc_byte)

        out_buffer = out_buffer[shift:]
        result = pickle.loads(bytes_list)
        results.append(result)

    return results
    

def any_broadcast(data, root_rank, n_gpu=None):
    """broadcast arbitrary data from root_rank to all nodes."""
    if n_gpu is None:
        n_gpu = torch.cuda.device_count()

    enc = pickle.dumps(data)

    tensor_list = [torch.zeros(1, dtype=torch.int64).cuda() for _ in range(n_gpu)]
    dist.all_gather(tensor_list, tensor=torch.tensor([len(enc)]).cuda())
    max_size = torch.cat(tensor_list, dim=0).view(-1).max().item()
    buffer_, enc_byte = _encode(enc, max_size, use_max_size=True)

    dist.broadcast(buffer_, root_rank)

    bytes_list, _ = _decode(buffer_, enc_byte)
    result = pickle.loads(bytes_list)
    return result
