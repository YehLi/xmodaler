"""
From original at https://github.com/facebookresearch/detectron2/blob/master/detectron2/config/defaults.py
Original copyright of Facebook code below, modifications by Yehao Li, Copyright 2021.	
"""
# Copyright (c) Facebook, Inc. and its affiliates.
from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# The version number, to upgrade from old configs to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
_C.VERSION = 1

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()

_C.DATASETS.TRAIN = ''

_C.DATASETS.VAL = ''

_C.DATASETS.TEST = ''

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()

_C.DATALOADER.TRAIN_BATCH_SIZE = 64

_C.DATALOADER.TEST_BATCH_SIZE = 64

_C.DATALOADER.NUM_WORKERS = 4

_C.DATALOADER.FEATS_FOLDER = ''

_C.DATALOADER.ANNO_FOLDER = ''

_C.DATALOADER.SEQ_PER_SAMPLE = 5

_C.DATALOADER.MAX_FEAT_NUM = -1

# -----------------------------------------------------------------------------
# Engine
# -----------------------------------------------------------------------------
_C.ENGINE = CN()

_C.ENGINE.NAME = 'DefaultTrainer'

# -----------------------------------------------------------------------------
# Scheduled sampling
# -----------------------------------------------------------------------------
_C.SCHEDULED_SAMPLING = CN()

_C.SCHEDULED_SAMPLING.START_EPOCH = 0

_C.SCHEDULED_SAMPLING.INC_EVERY_EPOCH = 5

_C.SCHEDULED_SAMPLING.INC_PROB = 0.05

_C.SCHEDULED_SAMPLING.MAX_PROB = 0.25

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()

_C.MODEL.DEVICE = "cuda"

_C.MODEL.VOCAB_SIZE = 1000 # include <BOS>/<EOS>

_C.MODEL.META_ARCHITECTURE = ''

_C.MODEL.ENCODER = ''

_C.MODEL.ENCODER_DIM = 1024

_C.MODEL.DECODER = ''

_C.MODEL.DECODER_DIM = 1024

_C.MODEL.PRED_DROPOUT = 0.0

_C.MODEL.PREDICTOR = ''

_C.MODEL.V_PREDICTOR = ''

_C.MODEL.MAX_SEQ_LEN = 17

_C.MODEL.WEIGHTS = ''

# ----------------------------------------------------------------------------
# Token embedding
# ----------------------------------------------------------------------------
_C.MODEL.TOKEN_EMBED = CN()

_C.MODEL.TOKEN_EMBED.NAME = ''

_C.MODEL.TOKEN_EMBED.DIM = 1024

_C.MODEL.TOKEN_EMBED.ACTIVATION = 'none'

_C.MODEL.TOKEN_EMBED.ELU_ALPHA = 0.5

_C.MODEL.TOKEN_EMBED.USE_NORM = False

_C.MODEL.TOKEN_EMBED.DROPOUT = 0.0

_C.MODEL.TOKEN_EMBED.POSITION = 'none'

_C.MODEL.TOKEN_EMBED.POSITION_MAX_LEN = 5000

_C.MODEL.TOKEN_EMBED.TYPE_VOCAB_SIZE = 0

# ----------------------------------------------------------------------------
# Visual embedding
# ----------------------------------------------------------------------------
_C.MODEL.VISUAL_EMBED = CN()

_C.MODEL.VISUAL_EMBED.NAME = ''

_C.MODEL.VISUAL_EMBED.IN_DIM = 2048

_C.MODEL.VISUAL_EMBED.OUT_DIM = 1024

_C.MODEL.VISUAL_EMBED.ACTIVATION = 'none'

_C.MODEL.VISUAL_EMBED.ELU_ALPHA = 0.5

_C.MODEL.VISUAL_EMBED.USE_NORM = False

_C.MODEL.VISUAL_EMBED.DROPOUT = 0.0

_C.MODEL.VISUAL_EMBED.LOCATION_SIZE = 0

# ----------------------------------------------------------------------------
# Pre-training
# ----------------------------------------------------------------------------
_C.MODEL.PRETRAINING = CN()

_C.MODEL.PRETRAINING.MODEL_NAME = 'bert-base-uncased'

_C.MODEL.PRETRAINING.FROM_PRETRAINED = 'bert-base-uncased'

_C.MODEL.PRETRAINING.DO_LOWER_CASE = True

# ----------------------------------------------------------------------------
# BERT
# ----------------------------------------------------------------------------
_C.MODEL.BERT = CN()

_C.MODEL.BERT.HIDDEN_SIZE = 512

_C.MODEL.BERT.HIDDEN_DROPOUT_PROB = 0.1

_C.MODEL.BERT.HIDDEN_ACT = "gelu"

_C.MODEL.BERT.NUM_ATTENTION_HEADS = 8

_C.MODEL.BERT.INTERMEDIATE_SIZE = 2048

_C.MODEL.BERT.INTERMEDIATE_DROP = 0.1

_C.MODEL.BERT.FFN_DROPOUT_PROB = 0.1

_C.MODEL.BERT.ATTENTION_PROBS_DROPOUT_PROB = 0.1

_C.MODEL.BERT.V_TARGET_SIZE = 0

_C.MODEL.BERT.NUM_HIDDEN_LAYERS = 12

_C.MODEL.BERT.V_NUM_HIDDEN_LAYERS = 6

_C.MODEL.BERT.NUM_UNDERSTANDING_LAYERS = 6

_C.MODEL.BERT.NUM_GENERATION_LAYERS = 6

# ----------------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------------
_C.SOLVER = CN()

_C.SOLVER.NAME = 'Adam'

_C.SOLVER.EPOCH = 10

_C.SOLVER.CHECKPOINT_PERIOD = 1

_C.SOLVER.EVAL_PERIOD = 1

_C.SOLVER.BASE_LR = 0.0005

_C.SOLVER.BIAS_LR_FACTOR = 1.0

_C.SOLVER.LR_DECAY = 0.0

_C.SOLVER.WEIGHT_DECAY = 0.0

_C.SOLVER.WEIGHT_DECAY_NORM = 0.0

_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0

_C.SOLVER.INITIAL_ACCUMULATOR_VALUE = 0.0

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.DAMPENING = 0.0

_C.SOLVER.NESTEROV = 0.0

_C.SOLVER.ALPHA = 0.99

_C.SOLVER.BETAS = [0.9, 0.999]

_C.SOLVER.EPS = 1e-8

_C.SOLVER.AMSGRAD = False

_C.SOLVER.CENTERED = False

_C.SOLVER.GRAD_CLIP_TYPE = 'norm' # norm, value

_C.SOLVER.GRAD_CLIP = 0.1

_C.SOLVER.NORM_TYPE = 2.0

_C.SOLVER.WRITE_PERIOD = 20

# ----------------------------------------------------------------------------
# lr scheduler
# ----------------------------------------------------------------------------
_C.LR_SCHEDULER = CN()

_C.LR_SCHEDULER.NAME = 'StepLR'

_C.LR_SCHEDULER.STEP_SIZE = 3

_C.LR_SCHEDULER.GAMMA = 0.1

_C.LR_SCHEDULER.MODEL_SIZE = -1 # for Noam only

_C.LR_SCHEDULER.FACTOR = 1.0 # for Noam only

_C.LR_SCHEDULER.WARMUP = 0 # epoch, for WarmupXXX; iteration, for Noam

_C.LR_SCHEDULER.MIN_LR = 0.00001 

_C.LR_SCHEDULER.MILESTONES = (3,) # for WarmupMultiStep only

_C.LR_SCHEDULER.WARMUP_FACTOR = 1.0 / 3 # for WarmupMultiStep only

_C.LR_SCHEDULER.WARMUP_METHOD = "linear" # for WarmupMultiStep only

# ---------------------------------------------------------------------------- #
# Losses
# ---------------------------------------------------------------------------- #
_C.LOSSES = CN()

_C.LOSSES.NAMES = ['']

_C.LOSSES.LABELSMOOTHING = 0.1

_C.LOSSES.MARGIN = 0.2

_C.LOSSES.MAX_VIOLATION = True

# ---------------------------------------------------------------------------- #
# SCORER options
# ---------------------------------------------------------------------------- #
_C.SCORER = CN()

_C.SCORER.NAME = ''

_C.SCORER.TYPES = ['']

_C.SCORER.WEIGHTS = [1.0]

_C.SCORER.GT_PATH = 'coco_train_gts.pkl'

_C.SCORER.CIDER_CACHED = 'coco_train_cider.pkl'

_C.SCORER.EOS_ID = 0

# ---------------------------------------------------------------------------- #
# Decode strategy
# ---------------------------------------------------------------------------- #
_C.DECODE_STRATEGY = CN()

_C.DECODE_STRATEGY.NAME = 'BeamSearcher'

_C.DECODE_STRATEGY.BEAM_SIZE = 1

# ---------------------------------------------------------------------------- #
# INFERENCE options
# ---------------------------------------------------------------------------- #
_C.INFERENCE = CN()

_C.INFERENCE.NAME = ''

_C.INFERENCE.VOCAB = 'coco_vocabulary.txt'

_C.INFERENCE.ID_KEY = 'image_id'

_C.INFERENCE.VALUE = 'caption'

_C.INFERENCE.VAL_ANNFILE = 'captions_val5k.json'

_C.INFERENCE.TEST_ANNFILE = 'captions_test5k.json'

_C.INFERENCE.GENERATION_MODE = True

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "./output"

_C.SEED = -1

_C.CUDNN_BENCHMARK = True

