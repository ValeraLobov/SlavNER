import os
import logging
from os.path import dirname
from datetime import datetime

import numpy as np
import yaml
from sklearn.metrics.pairwise import cosine_similarity
from addict import Dict
from transformers import XLMRobertaTokenizerFast, AutoTokenizer
import torch

BASEDIR = dirname(dirname(os.path.abspath(__file__)))

# os.environ["CUDA_VISIBLE_DEVICES"]="3"

if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
device = torch.device(dev)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(BASEDIR, "huawei_slavic_ner_models.log")),
        logging.StreamHandler()
    ]
)

with open(os.path.join(BASEDIR, 'config.yaml'), 'r') as file:
    config = Dict(yaml.safe_load(file))

current_ts = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
config.backbone_model.wechsel_c1_model_path = f"{config.backbone_model.wechsel_c1_model_path}_{current_ts}"
config.backbone_model.wechsel_c2_model_path = f"{config.backbone_model.wechsel_c2_model_path}_{current_ts}"
config.backbone_model.wechsel_c3_model_path = f"{config.backbone_model.wechsel_c3_model_path}_{current_ts}"

tag2id = {tag: id1 for id1, tag in enumerate(config.ner_task.unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}

tag_to_ix = tag2id
tag_to_ix.update({config.curriculum_learning_model.START_TAG: config.ner_task.num_labels, config.curriculum_learning_model.STOP_TAG: config.ner_task.num_labels+1})

if config.backbone_model.base == "xlm":
    tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')
elif config.backbone_model.base == "rubert":
    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
else:
    raise Exception("Model backbone is undefined. Please use 'xlm' or 'rubert' in config")


def compute_similarity(emb1, emb2):
    """
    compute cosine similarity between two Lists/np.arrays
    :param emb1: List or np.array
    :param emb2: List or np.array
    :return:
    """
    return cosine_similarity(np.array(emb1).reshape(1, -1), np.array(emb2).reshape(1, -1))[0][0]
