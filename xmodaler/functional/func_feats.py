import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

def pad_tensor(tensor, padding_value, use_mask):
    out = pad_sequence(tensor, batch_first=True, padding_value=padding_value)
    if use_mask:
        lengths = [t.size(0) for t in tensor]
        max_lengths = max(lengths)
        mask = torch.zeros((out.size(0), max_lengths), dtype=torch.float32)
        for i, length in enumerate(lengths):
            mask[i, 0:length] = 1
        return out, mask
    else:
        return out

def dict_to_cuda(input_dict):
    for key in input_dict:
        input_dict[key] = input_dict[key].cuda()

def dict_as_tensor(input_dict):
    for key in input_dict:
        if not isinstance(input_dict[key], str):
            input_dict[key] = torch.as_tensor(input_dict[key])

def boxes_to_locfeats(boxes, image_w, image_h):
    image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
    image_location[:, :4] = boxes
    image_location[:, 4] = (
        (image_location[:, 3] - image_location[:, 1])
        * (image_location[:, 2] - image_location[:, 0])
        / (float(image_w) * float(image_h))
    )

    image_location[:, 0] = image_location[:, 0] / float(image_w)
    image_location[:, 1] = image_location[:, 1] / float(image_h)
    image_location[:, 2] = image_location[:, 2] / float(image_w)
    image_location[:, 3] = image_location[:, 3] / float(image_h)
    return image_location

def iou(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.shape[0]
    K = gt_boxes.shape[0]

    gt_boxes_area = (
        (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
    ).reshape(1, K)

    anchors_area = (
        (anchors[:, 2] - anchors[:, 0] + 1) * (anchors[:, 3] - anchors[:, 1] + 1)
    ).reshape(N, 1)

    boxes = np.repeat(anchors.reshape(N, 1, 4), K, axis=1)
    query_boxes = np.repeat(gt_boxes.reshape(1, K, 4), N, axis=0)

    iw = (
        np.minimum(boxes[:, :, 2], query_boxes[:, :, 2])
        - np.maximum(boxes[:, :, 0], query_boxes[:, :, 0])
        + 1
    )
    iw[iw < 0] = 0

    ih = (
        np.minimum(boxes[:, :, 3], query_boxes[:, :, 3])
        - np.maximum(boxes[:, :, 1], query_boxes[:, :, 1])
        + 1
    )
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps