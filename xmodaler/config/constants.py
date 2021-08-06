# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""

from .config import CfgNode as CN

_K = CN()

kfg = _K

_K.IDS = 'IDS'

########################################## TOKENS ##########################################
_K.U_TOKENS_IDS = 'U_TOKENS_IDS'

_K.U_TOKENS_IDS_WO_MASK = 'U_TOKENS_IDS_WO_MASK'

_K.G_TOKENS_IDS = 'G_TOKENS_IDS'

_K.RELATION = 'RELATION'

_K.ATTRIBUTE = 'ATTRIBUTE'

_K.U_TOKENS_TYPE = 'U_TOKENS_TYPE'

_K.G_TOKENS_TYPE = 'G_TOKENS_TYPE'

_K.U_TOKEN_EMBED = 'U_TOKEN_EMBED'

_K.G_TOKEN_EMBED = 'G_TOKEN_EMBED'

########################################## MASKS ##########################################
_K.TOKENS_MASKS = 'TOKENS_MASKS'

_K.EXT_U_TOKENS_MASKS = 'EXT_U_TOKENS_MASKS'

_K.EXT_G_TOKENS_MASKS = 'EXT_G_TOKENS_MASKS'

########################################## TARGET ##########################################
_K.U_TARGET_IDS = 'U_TARGET_IDS'

_K.G_TARGET_IDS = 'G_TARGET_IDS'

_K.V_TARGET = 'V_TARGET'

_K.V_TARGET_LABELS = 'V_TARGET_LABELS'

_K.ITM_NEG_LABEL = 'ITM_NEG_LABEL'

########################################## FEATS ##########################################
_K.ATT_FEATS = 'ATT_FEATS'

_K.ATT_FEATS_WO_MASK = 'ATT_FEATS_WO_MASK'

_K.ATT_MASKS = 'ATT_MASKS'

_K.EXT_ATT_MASKS = 'EXT_ATT_MASKS'

_K.ATT_FEATS_LOC = 'ATT_FEATS_LOC'

_K.P_ATT_FEATS = 'P_ATT_FEATS'

_K.GLOBAL_FEATS = 'GLOBAL_FEATS'

########################################## STATES ##########################################
_K.U_HIDDEN_STATES = 'U_HIDDEN_STATES'

_K.G_HIDDEN_STATES = 'G_HIDDEN_STATES'

_K.HISTORY_STATES = 'HISTORY_STATES'

_K.ENC_HISTORY_STATES = 'ENC_HISTORY_STATES'

_K.U_CELL_STATES = 'U_CELL_STATES'

_K.G_CELL_STATES = 'G_CELL_STATES'

########################################## LOGITS ##########################################
_K.U_LOGITS = 'U_LOGITS'

_K.G_LOGITS = 'G_LOGITS'

_K.V_LOGITS = 'V_LOGITS'

_K.V_REGRESS = 'V_REGRESS'

_K.ITM_LOGITS = 'ITM_LOGITS'

########################################## Others ##########################################
_K.SEQ_PER_SAMPLE = 'SEQ_PER_SAMPLE'

_K.SAMPLE_PER_SAMPLE = 'SAMPLE_PER_SAMPLE'

_K.G_SENTS_IDS = 'G_SENTS_IDS'

_K.G_LOGP = 'G_LOGP'

_K.OUTPUT = 'OUTPUT'

_K.DECODE_BY_SAMPLE = 'DECODE_BY_SAMPLE'

_K.REWARDS = 'REWARDS'

_K.SS_PROB = 'SS_PROB'

_K.TIME_STEP = 'TIME_STEP'

_K.COCO_PATH = '../coco_caption'

_K.TEMP_DIR = './data/temp'